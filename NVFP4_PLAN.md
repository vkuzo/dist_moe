# NVFP4 Forward and Backward Plan

You’re right. Here are five separate logical graphs.

Legend:

```text
[P] persistent activation slot
[S] reusable scratch
[SYM] peer-addressable symmetric HBM
Q(A) packed NVFP4
A_bw = DQ(Q(A))
```

## 1. BF16 forward

```text
Local X BF16 + expert IDs
          │
          │ publish
          ▼
[SYM] BF16 X + routing IDs
          │
          ├──────────────→ routing counts and gather/scatter pointers
          │                                  │
          │                                  ▼
          │                           device forward planner
          │                           ┌───────┴────────┐
          │                           │                │
          │                       full-save        recompute
          │                           │                │
          │                           │         save local X [P]
          ▼                           │
┌─────────────────────┐              │
│ Gather + W13 GEMM   │              │
│                     │              │
│ X_gathered BF16     │──────────────┴──→ [P] or [S]
│ H1 BF16             │─────────────────→ [P] or [S]
└──────────┬──────────┘
           │
           ▼
       SwiGLU
           │
           ▼
       H2 BF16 ──────────────────────────→ [S] always
           │
           ▼
┌─────────────────────┐
│ W2 GEMM             │
│ + peer scatter      │
└──────────┬──────────┘
           │
           ▼
[SYM] H3 route outputs
           │
           ├── full-save ─────────────────→ H3 [P]
           │
           ▼
score × H3 + top-k reduction
           │
           ▼
       Y BF16
```

Persistent state:

```text
Full-save:  [X_gathered, H1, H3]
Recompute:  [original local X]

H2 is never persistent.
```

## 2. BF16 backward

```text
Saved routing + offsets + planner predicate
                      │
                      ▼
              device backward planner
                      │
          ┌───────────┴────────────┐
          │                        │
      full-save                 recompute
          │                        │
 load X_gathered, H1, H3      saved local X
          │                    → republish
          │                    → gather + W13
          │                    → recreate X_gathered, H1
          │                    → SwiGLU + W2
          │                    → recreate H3
          └───────────┬────────────┘
                      │
                      ▼
                    H1
                      │
                      │ SwiGLU, always rerun
                      ▼
                    H2 [S]
```

```text
grad_Y + scores + H3
           │
           ▼
  score/reduction backward
       │              │
       │              └────────────────────────────→ grad_scores
       ▼
    grad_H3
       │
       ├── W2 DGRAD with W2 ───────────────────────→ grad_H2
       │
       └── W2 WGRAD with H2 ───────────────────────→ grad_W2

grad_H2 + H1
       │
       ▼
SwiGLU backward
       │
       ▼
    grad_H1
       │
       ├── W13 DGRAD with W13 ─────────────────────→ route grad_X
       │                                                    │
       │                                             peer scatter
       │                                             + top-k sum
       │                                                    │
       │                                                    ▼
       │                                                 grad_X
       │
       └── W13 WGRAD with X_gathered ───────────────→ grad_W13
```

## 3. Current NVFP4 forward

```text
BF16 W13/W2                             prepared NVFP4 W13/W2
     │                                           │
     │ dynamic quantization                      │
     ▼                                           │
Q(W13), Q(W2) + scales [S] ◀─────────────────────┘
     │
     │
Local X BF16
     │
     │ per-token NVFP4 quantization
     ▼
[SYM] Q(X) + block scales + token outer scale
     │
     ├──────────────→ routing counts and gather/scatter pointers
     │
     ▼
bind inference scratch
     │
     ▼
┌──────────────────────────┐
│ NVFP4 W13                │
│                          │
│ Q(X) × Q(W13)            │
│ → H1 BF16                │─────────────────────→ [S]
└────────────┬─────────────┘
             │
             ▼
          SwiGLU
             │
             ▼
            H2
             │
             │ per-route NVFP4 quantization
             ▼
          Q(H2) + scales ─────────────────────────→ [S]
             │
             ▼
┌──────────────────────────┐
│ NVFP4 W2                 │
│                          │
│ Q(H2) × Q(W2)            │
│ → H3 BF16 + peer scatter │
└────────────┬─────────────┘
             │
             ▼
[SYM] H3 route outputs
             │
             ▼
score × H3 + top-k reduction
             │
             ▼
          Y BF16
```

After the current forward:

```text
Survives:     Y and caller-owned weights
Discarded:    dynamic Q(W), Q(X), H1, Q(H2), H3, routing state
Backward:     unsupported
```

## 4. Target NVFP4 forward

Training accepts BF16 weights only.

```text
BF16 W13/W2
     │
     │ dynamic quantization
     ▼
Q(W13), Q(W2) + scales [S]
     │
     │
Local X BF16 + expert IDs
     │
     │ per-token NVFP4 quantization and publication
     ▼
[SYM] Q(X) + scales
     │
     ├──────────────→ routing counts and gather/scatter pointers
     │                                  │
     │                                  ▼
     │                           device forward planner
     │                           ┌───────┴────────┐
     │                           │                │
     │                       full-save        recompute
     │                           │                │
     │                           │         save local X [P]
     ▼                           │
gather exact Q(X)                │
     │                           │
     ├── DQ ─────────────────────┴──→ X_bw [P] or [S]
     │
     ▼
┌──────────────────────────┐
│ NVFP4 W13                │
│                          │
│ Q(X) × Q(W13)            │
│ → BF16 H1 boundary       │──────────────→ H1 [P] or [S]
└────────────┬─────────────┘
             │
             ▼
          SwiGLU
             │
             ▼
            H2
             │
             │ per-route NVFP4 quantization
             ▼
          Q(H2) + scales ─────────────────→ [S] only
             │
             ▼
┌──────────────────────────┐
│ NVFP4 W2                 │
│                          │
│ Q(H2) × Q(W2)            │
│ → H3 BF16 + peer scatter │
└────────────┬─────────────┘
             │
             ▼
[SYM] H3 route outputs
             │
             ├── full-save ─────────────────→ H3 [P]
             │
             ▼
score × H3 + top-k reduction
             │
             ▼
          Y BF16
```

Persistent state:

```text
Full-save:  [X_bw = DQ(Q(X_gathered)), H1, H3]
Recompute:  [original local BF16 X]

H2 and Q(H2) are never persistent.
```

## 5. Target NVFP4 backward

Recreate QDQ weights:

```text
BF16 W13                            BF16 W2
    │                                   │
    │ quantize again                    │ quantize again
    ▼                                   ▼
Q(W13) + scales                     Q(W2) + scales
    │                                   │
    │ dequantize                        │ dequantize
    ▼                                   ▼
W13_bw = DQ(Q(W13))                 W2_bw = DQ(Q(W2))
```

Resolve activation state:

```text
Saved routing + offsets + planner predicate
                      │
                      ▼
              device backward planner
                      │
          ┌───────────┴────────────┐
          │                        │
      full-save                 recompute
          │                        │
   load X_bw, H1, H3          saved local X
          │                    → quantize/publish Q(X)
          │                    → recreate X_bw
          │                    → NVFP4 W13 → H1
          │                    → NVFP4 W2 → H3
          └───────────┬────────────┘
                      │
                      ▼
                    H1
                      │
                      │ SwiGLU
                      ▼
                 H2 BF16 [S]
                      │
                      │ same NVFP4 quantizer as forward
                      ▼
                 Q(H2) + scales [S]
                      │
                      │ dequantize
                      ▼
          H2_bw = DQ(Q(SwiGLU(H1))) [S]
```

Run the shared dense BF16 backward:

```text
grad_Y + scores + H3
           │
           ▼
  score/reduction backward
       │              │
       │              └────────────────────────────→ grad_scores
       ▼
    grad_H3
       │
       ├── BF16 W2 DGRAD with W2_bw ───────────────→ grad_H2
       │
       └── BF16 W2 WGRAD with H2_bw ───────────────→ grad_W2

grad_H2 + H1
       │
       ▼
SwiGLU backward
       │
       ▼
    grad_H1
       │
       ├── BF16 W13 DGRAD with W13_bw ─────────────→ route grad_X
       │                                                      │
       │                                               peer scatter
       │                                               + top-k sum
       │                                                      │
       │                                                      ▼
       │                                                   grad_X
       │
       └── BF16 W13 WGRAD with X_bw ─────────────────→ grad_W13
```

The target NVFP4 backward differs from BF16 only in operand preparation:

```text
BF16:   X_gathered, H2, W13, W2
NVFP4:  DQ(Q(X_gathered)),
        DQ(Q(SwiGLU(H1))),
        DQ(Q(W13)),
        DQ(Q(W2))
```
