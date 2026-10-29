# NVFP4 Forward and Backward Plan

The graphs below show the BF16 baseline, current NVFP4 implementation, and
target NVFP4 training implementation from left to right. Each column is a
vertical dataflow graph. Corresponding components are aligned in the same
table row.

Legend:

```text
[P] persistent activation slot
[S] reusable scratch
[SYM] peer-addressable symmetric HBM
Q(A) packed NVFP4
A_bw = DQ(Q(A))
```

## Forward

| BF16 forward | Current NVFP4 forward | Target NVFP4 forward |
| :---: | :---: | :---: |
| `BF16 W13/W2`<br>used directly by the GEMMs | `BF16 W13/W2 → dynamic Q(W13/W2) + scales [S]`<br>or caller-owned prepared NVFP4 weights | Training accepts `BF16 W13/W2` only<br>`BF16 W13/W2 → dynamic Q(W13/W2) + scales [S]` |
| ↓ | ↓ | ↓ |
| Local `X BF16 + expert IDs`<br>↓ publish<br>`[SYM] BF16 X + routing IDs` | Local `X BF16`<br>↓ per-token NVFP4 quantization<br>`[SYM] Q(X) + block scales + token outer scale` | Local `X BF16 + expert IDs`<br>↓ per-token NVFP4 quantization and publication<br>`[SYM] Q(X) + scales` |
| ↓ | ↓ | ↓ |
| Routing counts and gather/scatter pointers<br>↓<br>Device forward planner chooses full-save or recompute<br>Recompute saves local `X [P]` | Routing counts and gather/scatter pointers<br>↓<br>Bind inference scratch<br>No activation slot or saved backward state | Routing counts and gather/scatter pointers<br>↓<br>Device forward planner chooses full-save or recompute<br>Recompute saves local `X BF16 [P]` |
| ↓ | ↓ | ↓ |
| Gather BF16 source rows<br>↓<br>`X_gathered BF16`<br>`[P]` for full-save, `[S]` for recompute | Gather exact `Q(X)` rows for W13<br>No dense backward operand is retained | Gather exact `Q(X)` rows<br>↓ dequantize<br>`X_bw = DQ(Q(X_gathered))`<br>`[P]` for full-save, `[S]` for recompute |
| ↓ | ↓ | ↓ |
| **BF16 W13 GEMM**<br>`X_gathered × W13` | **NVFP4 W13 GEMM**<br>`Q(X) × Q(W13)` | **NVFP4 W13 GEMM**<br>`Q(X) × Q(W13)` |
| ↓ | ↓ | ↓ |
| `H1 BF16`<br>`[P]` for full-save, `[S]` for recompute | `H1 BF16 [S]` | `H1 BF16` boundary<br>`[P]` for full-save, `[S]` for recompute |
| ↓ | ↓ | ↓ |
| SwiGLU<br>↓<br>`H2 BF16 [S]`<br>H2 is never persistent | SwiGLU<br>↓<br>`H2`<br>↓ per-route NVFP4 quantization<br>`Q(H2) + scales [S]` | SwiGLU<br>↓<br>`H2`<br>↓ per-route NVFP4 quantization<br>`Q(H2) + scales [S]`<br>H2 and Q(H2) are never persistent |
| ↓ | ↓ | ↓ |
| **BF16 W2 GEMM + peer scatter**<br>`H2 × W2` | **NVFP4 W2 GEMM + peer scatter**<br>`Q(H2) × Q(W2)` | **NVFP4 W2 GEMM + peer scatter**<br>`Q(H2) × Q(W2)` |
| ↓ | ↓ | ↓ |
| `[SYM] H3` route outputs<br>Full-save retains `H3 [P]` | `[SYM] H3` route outputs<br>H3 is transient | `[SYM] H3` route outputs<br>Full-save retains `H3 [P]` |
| ↓ | ↓ | ↓ |
| Score × H3 + top-k reduction<br>↓<br>`Y BF16` | Score × H3 + top-k reduction<br>↓<br>`Y BF16` | Score × H3 + top-k reduction<br>↓<br>`Y BF16` |
| **Full-save:** `[X_gathered, H1, H3]`<br>**Recompute:** `[original local X]`<br>H2 remains scratch-only | **Survives:** Y and caller-owned weights<br>**Discarded:** dynamic Q(W), Q(X), H1, Q(H2), H3, and routing state<br>**Backward:** unsupported | **Full-save:** `[X_bw, H1, H3]`<br>**Recompute:** `[original local BF16 X]`<br>H2 and Q(H2) remain scratch-only |

## Backward

| BF16 backward | Current NVFP4 backward | Target NVFP4 backward |
| :---: | :---: | :---: |
| Saved routing, offsets, slot ID, planner predicate, scores, and BF16 weights | No routing or activation state is saved<br>Autograd is rejected | Saved routing, offsets, slot ID, planner predicate, scores, and logical BF16 weights |
| ↓ | ✕ | ↓ |
| Device backward planner<br>↓<br>**Full-save:** load `X_gathered, H1, H3`<br>**Recompute:** saved X → republish → gather + W13 → recreate `X_gathered, H1` → SwiGLU + W2 → recreate H3 | No backward planner or recomputation path | Device backward planner<br>↓<br>**Full-save:** load `X_bw, H1, H3`<br>**Recompute:** saved BF16 X → quantize/publish Q(X) → recreate `X_bw` → NVFP4 W13 → H1 → NVFP4 W2 → H3 |
| ↓ | ✕ | ↓ |
| Use original BF16 `W13` and `W2` directly | No backward weight operands | `BF16 W13/W2`<br>↓ quantize again<br>`Q(W13/W2) + scales`<br>↓ dequantize<br>`W13_bw = DQ(Q(W13))`<br>`W2_bw = DQ(Q(W2))` |
| ↓ | ✕ | ↓ |
| `H1`<br>↓ SwiGLU, always rerun<br>`H2 BF16 [S]` | H2 is not reconstructed | `H1`<br>↓ SwiGLU<br>`H2 BF16 [S]`<br>↓ same NVFP4 quantizer as forward<br>`Q(H2) + scales [S]`<br>↓ dequantize<br>`H2_bw = DQ(Q(SwiGLU(H1))) [S]` |
| ↓ | ✕ | ↓ |
| `grad_Y + scores + H3`<br>↓ score/reduction backward<br>`grad_H3` and `grad_scores` | No score or reduction backward | `grad_Y + scores + H3`<br>↓ score/reduction backward<br>`grad_H3` and `grad_scores` |
| ↓ | ✕ | ↓ |
| **W2 DGRAD:** `grad_H3 × W2 → grad_H2`<br>**W2 WGRAD:** `grad_H3ᵀ × H2 → grad_W2` | No W2 DGRAD or WGRAD | **BF16 W2 DGRAD:** `grad_H3 × W2_bw → grad_H2`<br>**BF16 W2 WGRAD:** `grad_H3ᵀ × H2_bw → grad_W2` |
| ↓ | ✕ | ↓ |
| `grad_H2 + H1`<br>↓ SwiGLU backward<br>`grad_H1` | No SwiGLU backward | `grad_H2 + H1`<br>↓ SwiGLU backward<br>`grad_H1` |
| ↓ | ✕ | ↓ |
| **W13 DGRAD:** `grad_H1 × W13 → route grad_X`<br>↓ peer scatter + top-k sum<br>`grad_X`<br><br>**W13 WGRAD:** `grad_H1ᵀ × X_gathered → grad_W13` | No W13 DGRAD or WGRAD | **BF16 W13 DGRAD:** `grad_H1 × W13_bw → route grad_X`<br>↓ peer scatter + top-k sum<br>`grad_X`<br><br>**BF16 W13 WGRAD:** `grad_H1ᵀ × X_bw → grad_W13` |
| ↓ | ✕ | ↓ |
| Return `grad_X`, `grad_scores`, `grad_W13`, and `grad_W2` | No gradients | Return `grad_X`, `grad_scores`, `grad_W13`, and `grad_W2` to the logical BF16 inputs |

The target NVFP4 backward differs from BF16 only in operand preparation:

| BF16 backward operand | Target NVFP4 backward operand |
| --- | --- |
| `X_gathered` | `DQ(Q(X_gathered))` |
| `H1` | Actual BF16 NVFP4 W13 output |
| `H2 = SwiGLU(H1)` | `DQ(Q(SwiGLU(H1)))`, reconstructed during backward |
| `H3` | Actual BF16 NVFP4 W2 output |
| `W13` | `DQ(Q(W13))`, reconstructed during backward |
| `W2` | `DQ(Q(W2))`, reconstructed during backward |
