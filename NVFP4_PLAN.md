# NVFP4 Forward and Backward Plan

The graphs are placed side by side below. Read from left to right to compare
the BF16 baseline, the current internally implemented NVFP4 training path,
and the completed target NVFP4 training path. Blank padding aligns
corresponding components.

Legend:

```text
[P] persistent activation slot
[S] reusable scratch
[SYM] peer-addressable symmetric HBM
Q(A) packed NVFP4
A_bw = DQ(Q(A))
```

Implementation status: the current NVFP4 training forward is implemented
behind the private registered-operation boundary. Its backward is unsupported,
and the public NVFP4 API remains inference-only. The final column shows the
completed training path after the future backward change.

## Forward

```text
BF16 FORWARD                                                             | CURRENT NVFP4 TRAINING FORWARD                                           | TARGET NVFP4 FORWARD
-------------------------------------------------------------------------+--------------------------------------------------------------------------+-------------------------------------------------------------------------
                                                                         | BF16 W13/W2                                                             | BF16 W13/W2
                                                                         |      │                                                                   |      │
                                                                         |      │ dynamic quantization                                              |      │ dynamic quantization
                                                                         |      ▼                                                                   |      ▼
                                                                         | Q(W13), Q(W2) + scales [temporary; not saved]                            | Q(W13), Q(W2) + scales [S]
                                                                         |      │                                                                   |      │
                                                                         |      │                                                                   |      │
Local X BF16 + expert IDs                                                | Local X BF16 + expert IDs                                                | Local X BF16 + expert IDs
          │                                                              |      │                                                                   |      │
          │ publish                                                      |      │ per-token NVFP4 quantization                                      |      │ per-token NVFP4 quantization and publication
          ▼                                                              |      ▼                                                                   |      ▼
[SYM] BF16 X + routing IDs                                               | [SYM] Q(X) + block scales + token outer scale                            | [SYM] Q(X) + scales
          │                                                              |      │                                                                   |      │
          ├──────────────→ routing counts and gather/scatter pointers    |      ├──────────────→ routing counts and gather/scatter pointers         |      ├──────────────→ routing counts and gather/scatter pointers
          │                                  │                           |      │                                                                   |      │                                  │
          │                                  ▼                           |      ▼                                                                   |      │                                  ▼
          │                           device forward planner             |      │                           hybrid device forward planner            |      │                           device forward planner
          │                           ┌───────┴────────┐                 |      │                           ┌───────┴────────┐                       |      │                           ┌───────┴────────┐
          │                           │                │                 |      │                           │                │                       |      │                           │                │
          │                       full-save        recompute             |      │                       full-save        recompute                   |      │                       full-save        recompute
          │                           │                │                 |      │                           │                │                       |      │                           │                │
          │                           │         save local X [P]         |      │                           │         save local X [P]               |      │                           │         save local X [P]
          ▼                           │                                  |      ▼                           │                                       |      ▼                           │
                                                                         | gather exact peer-addressed Q(X) │                                       | gather exact Q(X)                │
                                                                         |      │                           │                                       |      │                           │
                                                                         |      └── DQ ──→ X_bw [P]         │ no X_bw write                        |      ├── DQ ─────────────────────┴──→ X_bw [P] or [S]
                                                                         |                                  │                                       |      │
                                                                         |      ▼                           ▼                                       |      ▼
┌─────────────────────┐              │                                   | ┌──────────────────────────┐                                             | ┌──────────────────────────┐
│ Gather + W13 GEMM   │              │                                   | │ NVFP4 W13                │                                             | │ NVFP4 W13                │
│                     │              │                                   | │                          │                                             | │                          │
│ X_gathered BF16     │──────────────┴──→ [P] or [S]                     | │ Q(X) × Q(W13)            │                                             | │ Q(X) × Q(W13)            │
│ H1 BF16             │─────────────────→ [P] or [S]                     | │ → BF16 H1 boundary       │──────────────→ H1 [P] or [S]                | │ → BF16 H1 boundary       │──────────────→ H1 [P] or [S]
└──────────┬──────────┘                                                  | └────────────┬─────────────┘                                             | └────────────┬─────────────┘
           │                                                             |              │                                                           |              │
           ▼                                                             |              ▼                                                           |              ▼
       SwiGLU                                                            |           SwiGLU                                                         |           SwiGLU
           │                                                             |              │                                                           |              │
           ▼                                                             |              ▼                                                           |              ▼
       H2 BF16 ──────────────────────────→ [S] always                    |             H2                                                           |             H2
           │                                                             |              │                                                           |              │
           ▼                                                             |              │ per-route NVFP4 quantization                              |              │ per-route NVFP4 quantization
                                                                         |              ▼                                                           |              ▼
                                                                         |           Q(H2) + scales ─────────────────────────→ [S] only             |           Q(H2) + scales ─────────────────→ [S] only
                                                                         |              │                                                           |              │
                                                                         |              ▼                                                           |              ▼
┌─────────────────────┐                                                  | ┌──────────────────────────┐                                             | ┌──────────────────────────┐
│ W2 GEMM             │                                                  | │ NVFP4 W2                 │                                             | │ NVFP4 W2                 │
│ + peer scatter      │                                                  | │                          │                                             | │                          │
└──────────┬──────────┘                                                  | │ Q(H2) × Q(W2)            │                                             | │ Q(H2) × Q(W2)            │
           │                                                             | │ → H3 BF16 + peer scatter │                                             | │ → H3 BF16 + peer scatter │
           ▼                                                             | └────────────┬─────────────┘                                             | └────────────┬─────────────┘
                                                                         |              │                                                           |              │
                                                                         |              ▼                                                           |              ▼
[SYM] H3 route outputs                                                   | [SYM] H3 route outputs                                                   | [SYM] H3 route outputs
           │                                                             |              │                                                           |              │
           ├── full-save ─────────────────→ H3 [P]                       |              ├── full-save ─────────────────→ H3 [P]                      |              ├── full-save ─────────────────→ H3 [P]
           │                                                             |              │                                                           |              │
           ▼                                                             |              ▼                                                           |              ▼
score × H3 + top-k reduction                                             | score × H3 + top-k reduction                                             | score × H3 + top-k reduction
           │                                                             |              │                                                           |              │
           ▼                                                             |              ▼                                                           |              ▼
       Y BF16                                                            |           Y BF16                                                         |           Y BF16
                                                                         |                                                                          |
Full-save:  [X_gathered, H1, H3]                                         | Full-save:  [X_bw = DQ(Q(X_gathered)), H1, H3]                           | Full-save:  [X_bw = DQ(Q(X_gathered)), H1, H3]
Recompute:  [original local X]                                           | Recompute:  [original local BF16 X]                                      | Recompute:  [original local BF16 X]
                                                                         | Fixed 21-tensor state: 10 empty weight slots                             |
                                                                         |              + expert row counts                                        |
                                                                         |              + forward/backward gather and scatter pointers             |
                                                                         |              + per-rank row-count snapshot and local row count           |
                                                                         |              + need_recompute and all planner/descriptor offsets         |
                                                                         |              + device recompute predicate and postprocessing state       |
                                                                         | BF16 weights remain explicit registered-op inputs.                      |
                                                                         | Backward will re-quantize them.                                         |
                                                                         | Saved H1/H3 are the real NVFP4 GEMM BF16 boundaries.                    |
                                                                         | The fixed tensor schema is FakeTensor/make_fx/CUDA-graph visible.        |
                                                                         | Backward: unsupported                                                    |
H2 is never persistent.                                                  | H2, Q(H2), Q(X), and Q(W) are never persistent.                          | H2 and Q(H2) are never persistent.
```

## Backward

```text
BF16 BACKWARD                                                            | CURRENT NVFP4 TRAINING BACKWARD                                          | TARGET NVFP4 BACKWARD
-------------------------------------------------------------------------+--------------------------------------------------------------------------+-------------------------------------------------------------------------
                                                                         | Private forward producer now emits:                                      | BF16 W13                            BF16 W2
                                                                         |   full-save [X_bw, H1, H3]                                               |     │                                   │
                                                                         |   recompute [original local BF16 X]                                      |     │ quantize again                    │ quantize again
                                                                         |   fixed routing/planner/postprocessing state                             |     ▼                                   ▼
                                                                         |   no H2 or quantized tensor payloads                                     | Q(W13) + scales                     Q(W2) + scales
                                                                         |                                                                          |     │                                   │
                                                                         | DQ primitives available but not wired into backward:                     |     │ dequantize                        │ dequantize
                                                                         |   Q(A) + per-token outer scale  ── DQ ──→ BF16                           |     ▼                                   ▼
                                                                         |   Q(W) + per-expert outer scale ── DQ ──→ BF16                           | W13_bw = DQ(Q(W13))                 W2_bw = DQ(Q(W2))
                                                                         | Backward: unsupported                                                    |
Saved routing + offsets + planner predicate                              |                                                                          | Saved routing + offsets + planner predicate
                      │                                                  |                                                                          |                       │
                      ▼                                                  |                                                                          |                       ▼
              device backward planner                                    |                                                                          |               device backward planner
                      │                                                  |                                                                          |                       │
          ┌───────────┴────────────┐                                     |                                                                          |           ┌───────────┴────────────┐
          │                        │                                     |                                                                          |           │                        │
      full-save                 recompute                                |                                                                          |       full-save                 recompute
          │                        │                                     |                                                                          |           │                        │
 load X_gathered, H1, H3      saved local X                              |                                                                          |    load X_bw, H1, H3          saved local X
          │                    → republish                               |                                                                          |           │                    → quantize/publish Q(X)
          │                    → gather + W13                            |                                                                          |           │                    → recreate X_bw
          │                    → recreate X_gathered, H1                 |                                                                          |           │                    → NVFP4 W13 → H1
          │                    → SwiGLU + W2                             |                                                                          |           │                    → NVFP4 W2 → H3
          │                    → recreate H3                             |                                                                          |           └───────────┬────────────┘
          └───────────┬────────────┘                                     |                                                                          |                       │
                      │                                                  |                                                                          |                       ▼
                      ▼                                                  |                                                                          |
                    H1                                                   |                                                                          |                     H1
                      │                                                  |                                                                          |                       │
                      │ SwiGLU, always rerun                             |                                                                          |                       │ SwiGLU
                      ▼                                                  |                                                                          |                       ▼
                    H2 [S]                                               |                                                                          |                  H2 BF16 [S]
                                                                         |                                                                          |                       │
                                                                         |                                                                          |                       │ same NVFP4 quantizer as forward
                                                                         |                                                                          |                       ▼
                                                                         |                                                                          |                  Q(H2) + scales [S]
                                                                         |                                                                          |                       │
                                                                         |                                                                          |                       │ dequantize
                                                                         |                                                                          |                       ▼
                                                                         |                                                                          |           H2_bw = DQ(Q(SwiGLU(H1))) [S]
                                                                         |                                                                          |
grad_Y + scores + H3                                                     |                                                                          | grad_Y + scores + H3
           │                                                             |                                                                          |            │
           ▼                                                             |                                                                          |            ▼
  score/reduction backward                                               |                                                                          |   score/reduction backward
       │              │                                                  |                                                                          |        │              │
       │              └────────────────────────────→ grad_scores         |                                                                          |        │              └────────────────────────────→ grad_scores
       ▼                                                                 |                                                                          |        ▼
    grad_H3                                                              |                                                                          |     grad_H3
       │                                                                 |                                                                          |        │
       ├── W2 DGRAD with W2 ───────────────────────→ grad_H2             |                                                                          |        ├── BF16 W2 DGRAD with W2_bw ───────────────→ grad_H2
       │                                                                 |                                                                          |        │
       └── W2 WGRAD with H2 ───────────────────────→ grad_W2             |                                                                          |        └── BF16 W2 WGRAD with H2_bw ───────────────→ grad_W2
                                                                         |                                                                          |
grad_H2 + H1                                                             |                                                                          | grad_H2 + H1
       │                                                                 |                                                                          |        │
       ▼                                                                 |                                                                          |        ▼
SwiGLU backward                                                          |                                                                          | SwiGLU backward
       │                                                                 |                                                                          |        │
       ▼                                                                 |                                                                          |        ▼
    grad_H1                                                              |                                                                          |     grad_H1
       │                                                                 |                                                                          |        │
       ├── W13 DGRAD with W13 ─────────────────────→ route grad_X        |                                                                          |        ├── BF16 W13 DGRAD with W13_bw ─────────────→ route grad_X
       │                                                    │            |                                                                          |        │                                                      │
       │                                             peer scatter        |                                                                          |        │                                               peer scatter
       │                                             + top-k sum         |                                                                          |        │                                               + top-k sum
       │                                                    │            |                                                                          |        │                                                      │
       │                                                    ▼            |                                                                          |        │                                                      ▼
       │                                                 grad_X          |                                                                          |        │                                                   grad_X
       │                                                                 |                                                                          |        │
       └── W13 WGRAD with X_gathered ───────────────→ grad_W13           |                                                                          |        └── BF16 W13 WGRAD with X_bw ─────────────────→ grad_W13
```

The target NVFP4 backward differs from BF16 only in operand preparation:

```text
BF16:   X_gathered, H2, W13, W2
NVFP4:  DQ(Q(X_gathered)),
        DQ(Q(SwiGLU(H1))),
        DQ(Q(W13)),
        DQ(Q(W2))
```
