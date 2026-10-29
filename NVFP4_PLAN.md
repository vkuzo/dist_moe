# NVFP4 Forward and Backward Plan

The graphs are placed side by side below. Read from left to right to compare
the BF16 baseline, current NVFP4 implementation, and target NVFP4 training
implementation. Blank padding aligns corresponding components. The current
NVFP4 columns also distinguish available DQ primitives from integrated flow.

Legend:

```text
[P] persistent activation slot
[S] reusable scratch
[SYM] peer-addressable symmetric HBM
Q(A) packed NVFP4
A_bw = DQ(Q(A))
```

Implementation status: the target forward state producer is now implemented
behind the private registered-operation boundary. The public NVFP4 API remains
inference-only, and the target backward remains future work.

## Forward

```text
BF16 FORWARD                                                             | CURRENT NVFP4 FORWARD                                                    | TARGET NVFP4 FORWARD
-------------------------------------------------------------------------+--------------------------------------------------------------------------+-------------------------------------------------------------------------
                                                                         | BF16 W13/W2                             prepared NVFP4 W13/W2            | BF16 W13/W2
                                                                         |      │                                           │                       |      │
                                                                         |      │ dynamic quantization                      │                       |      │ dynamic quantization
                                                                         |      ▼                                           │                       |      ▼
                                                                         | Q(W13), Q(W2) + scales [S] ◀─────────────────────┘                       | Q(W13), Q(W2) + scales [S]
                                                                         |      │                                                                   |      │
                                                                         |      │                                                                   |      │
Local X BF16 + expert IDs                                                | Local X BF16                                                             | Local X BF16 + expert IDs
          │                                                              |      │                                                                   |      │
          │ publish                                                      |      │ per-token NVFP4 quantization                                      |      │ per-token NVFP4 quantization and publication
          ▼                                                              |      ▼                                                                   |      ▼
[SYM] BF16 X + routing IDs                                               | [SYM] Q(X) + block scales + token outer scale                            | [SYM] Q(X) + scales
          │                                                              |      │                                                                   |      │
          ├──────────────→ routing counts and gather/scatter pointers    |      ├──────────────→ routing counts and gather/scatter pointers         |      ├──────────────→ routing counts and gather/scatter pointers
          │                                  │                           |      │                                                                   |      │                                  │
          │                                  ▼                           |      ▼                                                                   |      │                                  ▼
          │                           device forward planner             | bind inference scratch                                                   |      │                           device forward planner
          │                           ┌───────┴────────┐                 |      │                                                                   |      │                           ┌───────┴────────┐
          │                           │                │                 |      ▼                                                                   |      │                           │                │
          │                       full-save        recompute             |                                                                          |      │                       full-save        recompute
          │                           │                │                 |                                                                          |      │                           │                │
          │                           │         save local X [P]         |                                                                          |      │                           │         save local X [P]
          ▼                           │                                  |                                                                          |      ▼                           │
                                                                         |                                                                          | gather exact Q(X)                │
                                                                         |                                                                          |      │                           │
                                                                         |                                                                          |      ├── DQ ─────────────────────┴──→ X_bw [P] or [S]
                                                                         |                                                                          |      │
                                                                         |                                                                          |      ▼
┌─────────────────────┐              │                                   | ┌──────────────────────────┐                                             | ┌──────────────────────────┐
│ Gather + W13 GEMM   │              │                                   | │ NVFP4 W13                │                                             | │ NVFP4 W13                │
│                     │              │                                   | │                          │                                             | │                          │
│ X_gathered BF16     │──────────────┴──→ [P] or [S]                     | │ Q(X) × Q(W13)            │                                             | │ Q(X) × Q(W13)            │
│ H1 BF16             │─────────────────→ [P] or [S]                     | │ → H1 BF16                │─────────────────────→ [S]                   | │ → BF16 H1 boundary       │──────────────→ H1 [P] or [S]
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
                                                                         |           Q(H2) + scales ─────────────────────────→ [S]                  |           Q(H2) + scales ─────────────────→ [S] only
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
           ├── full-save ─────────────────→ H3 [P]                       |              ▼                                                           |              ├── full-save ─────────────────→ H3 [P]
           │                                                             | score × H3 + top-k reduction                                             |              │
           ▼                                                             |              │                                                           |              ▼
score × H3 + top-k reduction                                             |              ▼                                                           | score × H3 + top-k reduction
           │                                                             |           Y BF16                                                         |              │
           ▼                                                             |                                                                          |              ▼
       Y BF16                                                            |                                                                          |           Y BF16
                                                                         |                                                                          |
Full-save:  [X_gathered, H1, H3]                                         | Survives:     Y and caller-owned weights                                 | Full-save:  [X_bw = DQ(Q(X_gathered)), H1, H3]
Recompute:  [original local X]                                           | Discarded:    dynamic Q(W), Q(X), H1, Q(H2), H3, routing state           | Recompute:  [original local BF16 X]
                                                                         | Backward:     unsupported                                                |
                                                                         | DQ available: exact Q(A) per-token and Q(W) per-expert → BF16            |
                                                                         | Buffer form: device offset + predicate; not yet integrated               |
                                                                         | Integration:  not called by the current inference path                   |
H2 is never persistent.                                                  |                                                                          | H2 and Q(H2) are never persistent.
```

## Backward

```text
BF16 BACKWARD                                                            | CURRENT NVFP4 BACKWARD                                                   | TARGET NVFP4 BACKWARD
-------------------------------------------------------------------------+--------------------------------------------------------------------------+-------------------------------------------------------------------------
                                                                         | DQ primitives available but not wired into backward:                     | BF16 W13                            BF16 W2
                                                                         |   Q(A) + per-token outer scale  ── DQ ──→ BF16                           |     │                                   │
                                                                         |   Q(W) + per-expert outer scale ── DQ ──→ BF16                           |     │ quantize again                    │ quantize again
                                                                         |                                                                          |     ▼                                   ▼
                                                                         | Backward: unsupported                                                    | Q(W13) + scales                     Q(W2) + scales
                                                                         |                                                                          |     │                                   │
                                                                         |                                                                          |     │ dequantize                        │ dequantize
                                                                         |                                                                          |     ▼                                   ▼
                                                                         |                                                                          | W13_bw = DQ(Q(W13))                 W2_bw = DQ(Q(W2))
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
