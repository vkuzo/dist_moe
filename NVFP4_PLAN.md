# NVFP4 Forward and Backward

The graphs compare the BF16 baseline with the implemented NVFP4 training
path. Corresponding components are aligned horizontally.

Legend:

```text
[P] persistent activation slot
[S] reusable scratch
[SYM] peer-addressable symmetric HBM
q(a_bf16) -> a_nvfp4: packed qdata and every required scale
a_q_dq_bf16 = dq(a_nvfp4)
```

## Forward

```text
BF16 FORWARD                                                             | IMPLEMENTED NVFP4 TRAINING FORWARD
-------------------------------------------------------------------------+--------------------------------------------------------------------------------
Inputs: x_bf16, w13_bf16, w2_bf16, topk_ids, topk_scores, slot_id        | Inputs: x_bf16, w13_bf16, w2_bf16, topk_ids, topk_scores, slot_id
                                                                         | w13_bf16, w2_bf16
                                                                         |      │
                                                                         |      │ dynamic per-expert quantization
                                                                         |      ▼
                                                                         | w13_nvfp4 = q(w13_bf16), w2_nvfp4 = q(w2_bf16) [S; not saved]
                                                                         |
x_bf16 + topk_ids                                                        | x_bf16 + topk_ids
          │                                                              |      │
          │ copy_routing_and_dispatch                                    |      ├── copy routing
          │                                                              |      └── quantize/publish x
          ▼                                                              |      ▼
dispatch_x_bf16 [SYM]                                                    | x_nvfp4 = q(x_bf16) [SYM]
routing_ids, routing_token_count                                         | routing_ids, routing_token_count
          │                                                              |      │
          └──────────── dispatch barrier ────────────┐                   |      └──────────── dispatch barrier ────────────┐
                                                    ▼                    |                                                    ▼
                                         dist_dispatch_routing           |                                         dist_dispatch_routing
                                                    │                    |                                                    │
                                                    ├── expert_counts    |                                                    ├── expert_counts
                                                    ├── recv_by_rank     |                                                    ├── recv_by_rank
                                                    ├── recv_count       |                                                    ├── recv_count
                                                    └── fwd_gather_ptrs, |                                                    └── fwd_gather_ptrs,
                                                        bwd_gather_ptrs, |                                                        bwd_gather_ptrs,
                                                        scatter_ptrs     |                                                        scatter_ptrs
                                                    │                    |                                                    │
                                                    ▼                    |                                                    ▼
                                         device forward planner          |                                         hybrid device forward planner
                                         inputs: recv_count,             |                                         inputs: recv_count,
                                         recv_by_rank, slot_id           |                                         recv_by_rank, slot_id
                                                    │                    |                                                    │
                           need_recompute, forward_state, recv_snapshot  |     need_recompute, recompute_condition, forward_offsets,
                                                    │                    |     activation_offsets, recv_snapshot
                                  ┌─────────────────┴──────────────┐     |                                  ┌─────────────────┴──────────────┐
                                  │                                │     |                                  │                                │
                              full-save                        recompute |                              full-save                        recompute
                                  │                                │     |                                  │                                │
                                  │                  x_saved_bf16 [P]    |                                  │                  x_saved_bf16 [P]
                                  │                                │     |                                  │                                │
                                  ▼                                ▼     |                                  ▼                                ▼
┌──────────────────────────────────────────────────────────────┐         | ┌──────────────────────────────────────────────────────────────┐
│ KERNEL: peer gather + BF16 W13 GEMM                          │         | │ KERNEL: staged NVFP4 W13 GEMM + packed-X copy                │
│ inputs: dispatch_x_bf16, fwd_gather_ptrs, expert_counts,     │         | │ inputs: x_nvfp4 [SYM], w13_nvfp4                              │
│         w13_bf16                                             │         | │         fwd_gather_ptrs, expert_counts                        │
│ outputs: x_gathered_bf16, h1_bf16                            │         | │ outputs: x_gathered_nvfp4, h1_bf16                           │
│ default presets also fuse SwiGLU and output h2_bf16          │         | └──────────────────────────────┬───────────────────────────────┘
└──────────────────────────────┬───────────────────────────────┘         |                                │
                               ├── x_gathered_bf16 [P] or [S]            |                                ├── x_gathered_nvfp4 [P] or [S]
                               ├── h1_bf16 [P] or [S]                    |                                └── h1_bf16 [P] or [S]
                               └── h2_bf16 [S]                           |                                             ▼
                                   (otherwise standalone SwiGLU)         | ┌──────────────────────────────────────────────────────────────┐
                                             │                           | │ ONE KERNEL: q(SwiGLU(h1_bf16)) + W2 GEMM + peer scatter       │
                                             ▼                           | │ inputs: h1_bf16, w2_nvfp4, expert_counts, scatter_ptrs        │
┌──────────────────────────────────────────────────────────────┐         | │ outputs: h2_nvfp4 [S], h3_route_bf16 [SYM]                    │
│ KERNEL: BF16 W2 GEMM + peer scatter                          │         | └──────────────────────────────┬───────────────────────────────┘
│ inputs: h2_bf16, w2_bf16, expert_counts, scatter_ptrs        │         |                                │
│ output: h3_route_bf16 [SYM]                                  │         |                                │
└──────────────────────────────┬───────────────────────────────┘         |                                │
                               ▼                                         |                                ▼
h3_route_bf16 [SYM] ── combine barrier                                   | h3_route_bf16 [SYM] ── combine barrier
          │                                                              |      │
          ▼                                                              |      ▼
score × h3_route_bf16 + top-k reduction                                  | score × h3_route_bf16 + top-k reduction
inputs: h3_route_bf16, topk_scores                                       | inputs: h3_route_bf16, topk_scores
outputs: y_bf16; full-save also writes h3_saved_bf16 [P]                 | outputs: y_bf16; full-save also writes h3_saved_bf16 [P]
                                                                         |
Full-save: [x_gathered_bf16, h1_bf16, h3_saved_bf16]                     | Full-save: [x_gathered_nvfp4, h1_bf16, h3_saved_bf16]
Recompute: [x_saved_bf16]                                                | Recompute: [x_saved_bf16]
                                                                         |
Both save: weights, topk_ids/scores, expert_counts, recv_count,          | Fixed state retains every named metadata tensor:
fwd_gather_ptrs, bwd_gather_ptrs, scatter_ptrs, recv_snapshot,           | expert_counts, fwd_gather_ptrs, bwd_gather_ptrs, scatter_ptrs,
need_recompute, forward_state, slot and postprocess state.               | recv_count, recv_snapshot, need_recompute, recompute_condition,
                                                                         | forward_offsets, activation_offsets, slot and postprocess state.
                                                                         | Native BF16 weights and topk_ids/scores are also retained.
                                                                         |
h2_bf16 is never persistent.                                             | h2_nvfp4, published x_nvfp4, w13_nvfp4, and w2_nvfp4 are transient.
```

## Backward

```text
BF16 BACKWARD                                                            | IMPLEMENTED NVFP4 TRAINING BACKWARD
-------------------------------------------------------------------------+--------------------------------------------------------------------------------
Inputs: grad_y_bf16 and all saved forward tensors                        | Same grad_y_bf16 and saved NVFP4-forward state
Native w13_bf16, w2_bf16                                                 | Native w13_bf16, w2_bf16
                                                                         |       │ q(w13_bf16), q(w2_bf16)
                                                                         |       ▼
                                                                         | w13_nvfp4, w2_nvfp4 [S]
                                                                         |       │ dq(w13_nvfp4), dq(w2_nvfp4)
                                                                         |       ▼
                                                                         | w13_q_dq_bf16, w2_q_dq_bf16 [S]
                                                                         |
Planner inputs: need_recompute, forward_state, recv_count,               | Planner inputs: need_recompute, recompute_condition,
recv_snapshot                                                            | forward_offsets, activation_offsets, recv_count,
                                                                         | recv_snapshot, slot_id
                         │                                               |                          │
                         ▼                                               |                          ▼
               device backward planner                                   |                device backward planner
                         │                                               |                          │
             backward_offsets                                            |       backward_offsets, backward_activation_offsets
                         │                                               |                          │
             ┌───────────┴────────────┐                                  |              ┌───────────┴────────────┐
             │                        │                                  |              │                        │
         full-save                 recompute                             |          full-save                 recompute
             │                        │                                  |              │                        │
 x_gathered_bf16 [P]          x_saved_bf16 [P]                           | x_gathered_nvfp4 [P]         x_saved_bf16 [P]
 h1_bf16 [P]                       │                                     | h1_bf16 [P]                       │
 h3_saved_bf16 [P]                 ├── publish                           | h3_saved_bf16 [P]                 ├── conditional load
             │                     ▼                                     |              │                     ▼
             │             dispatch_x_bf16 [SYM]                         |              │             local_x_bf16
             │                     │ dispatch barrier                    |              │                     │ q(local_x_bf16)/publish
             │                     ▼                                     |              │                     ▼
             │       BF16 W13 reconstruction kernel                      |              │             x_nvfp4 [SYM]
             │ inputs: dispatch_x_bf16, fwd_gather_ptrs,                 |              │                     │ dispatch barrier
             │         expert_counts, w13_bf16                           |              │                     ▼
             │ outputs: x_gathered_recomputed_bf16 [S],                  |              │        NVFP4 W13 reconstruction kernel
             │          h1_recomputed_bf16 [S]                           |              │ inputs: x_nvfp4, fwd_gather_ptrs,
             └──────────┬─────────────────────┘                          |              │         expert_counts, w13_nvfp4
                        │                                                |              │ outputs: x_gathered_recomputed_nvfp4 [S],
                        ▼                                                |              │          h1_recomputed_bf16 [S]
select x_for_bwd_bf16, h1_for_bwd_bf16                                   |              │                     │
                                                                         |              └─────────────┬────────────────┘
                        │                                                |                            │
                        │                                                |                            ▼
                        │                                                | select x_gathered_for_bwd_nvfp4, h1_for_bwd_bf16
                        │                                                |                            │
                        │                                                | KERNEL: dq(x_gathered_for_bwd_nvfp4)
                        │                                                | input: x_gathered_for_bwd_nvfp4 [P] or [S]
                        │                                                | output: x_q_dq_for_bwd_bf16 [S]
                        │                                                |                            │
                        │ standalone SwiGLU (always rerun)               |                            ▼
                        ▼                                                | x_q_dq_for_bwd_bf16 [S] + h1_for_bwd_bf16
                    h2_bf16 [S]                                          |                            │
                        │                                                |                            │ ONE KERNEL: q(SwiGLU) + W2 GEMM + peer scatter
                        │ KERNEL: conditional BF16 W2 + peer scatter     |                            │ inputs: h1_for_bwd_bf16, w2_nvfp4,
                        │ inputs: h2_bf16, w2_bf16,                      |                            │         expert_counts, scatter_ptrs
                        │         expert_counts, scatter_ptrs            |                            │ outputs:
                        ▼                                                |                            ▼
             h3_candidate_bf16 [SYM]                                     | h2_nvfp4 = q(SwiGLU(h1_for_bwd_bf16)) [S]
                                                                         | h3_candidate_bf16 [SYM]
                                                                         |                     │                 │
                        │ conditional combine barrier                    |                     │ dq(h2_nvfp4)   │ adapter combine barrier
                        ▼                                                |                     ▼                 ▼
select h3_for_bwd_bf16 = h3_saved or h3_candidate                        |          h2_q_dq_bf16 [S]    h3_candidate peer-visible
                                                                         |                            │
                                                                         |                            ▼
                                                                         | select h3_for_bwd_bf16 = h3_saved or h3_candidate
                                                                         |
grad_y_bf16 + topk_scores + h3_for_bwd_bf16                              | grad_y_bf16 + topk_scores + h3_for_bwd_bf16
                         │                                               |                          │
                         ▼                                               |                          ▼
              score/reduction backward                                   |               shared BF16 score/reduction backward
                         │                                               |                          │
              ┌──────────┴───────────┐                                   |               ┌──────────┴───────────┐
              ▼                      ▼                                   |               ▼                      ▼
grad_h3_route_bf16 [SYM]       grad_scores [OUT]                         | grad_h3_route_bf16 [SYM]       grad_scores [OUT]
              │ combine barrier                                          |               │ combine barrier
              ▼                                                          |               ▼
┌──────────────────────────────────────────────────────────────┐         | ┌──────────────────────────────────────────────────────────────┐
│ KERNEL: BF16 W2 DGRAD + peer gather                          │         | │ KERNEL: shared BF16 W2 DGRAD + peer gather                   │
│ inputs: grad_h3_route_bf16, scatter_ptrs, expert_counts      │         | │ inputs: grad_h3_route_bf16, scatter_ptrs, expert_counts       │
│         w2_bf16                                              │         | │         w2_q_dq_bf16                                         │
│ outputs: grad_h3_gathered_bf16 [S], grad_h2_bf16 [S]         │         | │ outputs: grad_h3_gathered_bf16 [S], grad_h2_bf16 [S]          │
└──────────────────────────────────────────────────────────────┘         | └──────────────────────────────────────────────────────────────┘
              │                                                          |               │
              ├── W2 WGRAD inputs: grad_h3_gathered_bf16, h2_bf16,       |               ├── W2 WGRAD inputs: grad_h3_gathered_bf16,
              │                    expert_counts                         |               │                    h2_q_dq_bf16, expert_counts
              │                 output: grad_w2                          |               │                 output: grad_w2
              ▼                                                          |               ▼
┌──────────────────────────────────────────────────────────────┐         | ┌──────────────────────────────────────────────────────────────┐
│ KERNEL when eligible: SwiGLU backward + BF16 W13 DGRAD       │         | │ KERNEL when eligible: shared SwiGLU backward + BF16 W13 DGRAD│
│ inputs: grad_h2_bf16, h1_for_bwd_bf16, w13_bf16             │          | │ inputs: grad_h2_bf16, h1_for_bwd_bf16, w13_q_dq_bf16         │
│         bwd_gather_ptrs                                      │         | │         bwd_gather_ptrs                                      │
│ outputs: grad_h1_bf16 [S], grad_x_route_bf16 [SYM]           │         | │ outputs: grad_h1_bf16 [S], grad_x_route_bf16 [SYM]            │
│ fallback: standalone SwiGLU backward, then W13 DGRAD         │         | │ fallback: same two separate BF16 kernels                      │
└──────────────────────────────────────────────────────────────┘         | └──────────────────────────────────────────────────────────────┘
              │                                                          |               │
              ├── W13 WGRAD inputs: grad_h1_bf16, x_for_bwd_bf16,        |               ├── W13 WGRAD inputs: grad_h1_bf16,
              │                     expert_counts                        |               │                     x_q_dq_for_bwd_bf16, expert_counts
              │                  output: grad_w13                        |               │                  output: grad_w13
              ▼                                                          |               ▼
grad_x_route_bf16 [SYM] ── dispatch barrier                              | grad_x_route_bf16 [SYM] ── dispatch barrier
              │                                                          |               │
              │ reduce_from_topk(topk_ids)                               |               │ shared BF16 reduce_from_topk(topk_ids)
              ▼                                                          |               ▼
grad_x_bf16 [OUT]                                                        | grad_x_bf16 [OUT]
                                                                         |
Outputs: grad_x_bf16, grad_scores, grad_w13, grad_w2                     | Outputs: grad_x_bf16, grad_scores, grad_w13, grad_w2; slot rewound
```

At the shared BF16 gradient-math boundary, NVFP4 supplies these reconstructed
operands:

```text
BF16:   x_for_bwd_bf16, h2_bf16, w13_bf16, w2_bf16
NVFP4:  x_q_dq_for_bwd_bf16,
        h2_q_dq_bf16,
        w13_q_dq_bf16,
        w2_q_dq_bf16
```
