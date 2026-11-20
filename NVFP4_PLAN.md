# NVFP4 Forward and Backward

The graphs document the BF16 baseline and the implemented NVFP4 training
path. Each execution path is shown in its own vertical section.

Legend:

```text
[P] persistent activation slot
[S] reusable scratch
[SYM] peer-addressable symmetric HBM
M_dense: routed-row capacity before per-expert NVFP4 kernel padding
M_pad: routed-row capacity after per-expert NVFP4 kernel padding
q(a_bf16) -> a_nvfp4: packed qdata and every required scale
a_q_dq_bf16 = dq(a_nvfp4)
```

## BF16

### Forward

```text
BF16 FORWARD
-----------------------------------------------------------------------------------------
Inputs: x_bf16, w13_bf16, w2_bf16, topk_ids, topk_scores, slot_id











┌──────────────────────────────────────────────────────────────┐
│ KERNEL: copy_routing_and_dispatch                            │
│ inputs: x_bf16, topk_ids                                     │
│ outputs: dispatch_x_bf16 [SYM], routing_ids,                 │
│          routing_token_count                                 │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               │
                               │
                               │
                               └─ dispatch barrier ─┐
                                                    ▼
                     ┌──────────────────────────────────────────────────────────────┐
                     │ KERNEL: dist_dispatch_routing                                │
                     │ inputs: routing_ids, routing_token_count                     │
                     │ outputs: expert_counts, recv_by_rank, recv_count,            │
                     │          fwd_gather_ptrs, bwd_gather_ptrs, scatter_ptrs      │
                     └──────────────────────────────┬───────────────────────────────┘
                                                    ▼
                     ┌──────────────────────────────────────────────────────────────┐
                     │ KERNEL: device forward planner                               │
                     │ inputs: recv_count, recv_by_rank, slot_id                    │
                     │ outputs: need_recompute, forward_state, recv_snapshot        │
                     │                                                              │
                     └──────────────────────────────┬───────────────────────────────┘
                                  ┌─────────────────┴──────────────┐
                                  │                                │
                              full-save                        recompute
                                  │                                │
                                  │                  x_saved_bf16 [P]
                                  │                                │
                                  ▼                                ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: peer gather + BF16 W13 GEMM                          │
│ inputs: dispatch_x_bf16, fwd_gather_ptrs, expert_counts,     │
│         w13_bf16                                             │
│ outputs: x_gathered_bf16, h1_bf16; default presets also      │
│          fuse SwiGLU and output h2_bf16                      │
└──────────────────────────────┬───────────────────────────────┘
                               ├── x_gathered_bf16 [P] or [S]
                               ├── h1_bf16 [P] or [S]
                               └── h2_bf16 [S] when fused
                                   otherwise:
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: standalone BF16 SwiGLU                               │
│ inputs: h1_bf16                                              │
│ outputs: h2_bf16 [S]                                         │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼

┌──────────────────────────────────────────────────────────────┐
│ KERNEL: BF16 W2 GEMM + peer scatter                          │
│ inputs: h2_bf16, w2_bf16, expert_counts, scatter_ptrs        │
│ outputs: h3_route_bf16 [SYM]                                 │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
h3_route_bf16 [SYM] ── combine barrier
          │
          ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: score × h3_route_bf16 + top-k reduction              │
│ inputs: h3_route_bf16, topk_scores                           │
│ outputs: y_bf16; full-save also writes h3_saved_bf16 [P]     │
└──────────────────────────────────────────────────────────────┘

Full-save: [x_gathered_bf16 [M_dense,D], h1_bf16 [M_dense,2F],
            h3_saved_bf16 [T,K,D]]
Recompute: [x_saved_bf16 [T,D]]

Both save: weights, topk_ids/scores, expert_counts, recv_count,
fwd_gather_ptrs, bwd_gather_ptrs, scatter_ptrs, recv_snapshot,
need_recompute, forward_state, slot and postprocess state.



h2_bf16 is never persistent.


```

### Backward

```text
BF16 BACKWARD
-----------------------------------------------------------------------------------------
Inputs: grad_y_bf16 and all saved forward tensors
Native w13_bf16, w2_bf16












┌──────────────────────────────────────────────────────────────┐
│ KERNEL: device backward planner                              │
│ inputs: need_recompute, forward_state, recv_count,           │
│         recv_snapshot                                        │
│                                                              │
│ outputs: backward_offsets                                    │
└──────────────────────────────┬───────────────────────────────┘
             ┌───────────┴────────────┐
             │                        │
         full-save                 recompute
             │                        │
 x_gathered_bf16 [P,M_dense,D] x_saved_bf16 [P,T,D]
 h1_bf16 [P,M_dense,2F]            │
 h3_saved_bf16 [P]                 ▼
             │       ┌───────────────────────────────────────────────┐
             │       │ KERNEL: publish saved X                       │
             │       │ inputs: x_saved_bf16 [P]                      │
             │       │ outputs: dispatch_x_bf16 [SYM]                │
             │       └───────────────────────┬───────────────────────┘
             │                               │
             │                               │
             │                               │
             │                               │
             │                               │
             │                               │ dispatch barrier
             │                               ▼
             │  ┌──────────────────────────────────────────────────────┐
             │  │ KERNEL: BF16 W13 reconstruction                      │
             │  │ inputs: dispatch_x_bf16, fwd_gather_ptrs,            │
             │  │         expert_counts, w13_bf16                      │
             │  │ outputs: x_gathered_recomputed_bf16 [S],             │
             │  │          h1_recomputed_bf16 [S]                      │
             │  └──────────────────────────┬───────────────────────────┘
             └─────────────────────────────┤
                                           ▼
select x_for_bwd_bf16, h1_for_bwd_bf16
                        │
                        │
                        │
                        │
                        │
                        │
                        │
                        │
                        │
                        │
                        │
                        │
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: standalone SwiGLU                                    │
│ inputs: h1_for_bwd_bf16                                      │
│ outputs: h2_bf16 [S]                                         │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: conditional BF16 W2 GEMM + peer scatter              │
│ inputs: h2_bf16, w2_bf16, expert_counts, scatter_ptrs        │
│ outputs: h3_candidate_bf16 [SYM]                             │
└──────────────────────────────┬───────────────────────────────┘
                               │ conditional combine barrier
                               │
                               │
                               ▼
select h3_for_bwd_bf16 = h3_saved or h3_candidate






┌──────────────────────────────────────────────────────────────┐
│ KERNEL: score/reduction backward                             │
│ inputs: grad_y_bf16, topk_scores, h3_for_bwd_bf16            │
│ outputs: grad_h3_route_bf16 [SYM], grad_scores [OUT]         │
└──────────────────────────────┬───────────────────────────────┘
              │ combine barrier
              │
              │
              │
              │
              │
              │
              ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: BF16 W2 DGRAD + peer gather                          │
│ inputs: grad_h3_route_bf16, scatter_ptrs, expert_counts,     │
│         w2_bf16                                              │
│ outputs: grad_h3_gathered_bf16 [S], grad_h2_bf16 [S]         │
└──────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: BF16 W2 WGRAD                                        │
│ inputs: grad_h3_gathered_bf16, h2_bf16, expert_counts        │
│ outputs: grad_w2                                             │
└──────────────────────────────┬───────────────────────────────┘
              │
              │
              │
              │
              │
              │
              ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: SwiGLU backward + BF16 W13 DGRAD, when eligible      │
│ inputs: grad_h2_bf16, h1_for_bwd_bf16, w13_bf16,             │
│         bwd_gather_ptrs                                      │
│ outputs: grad_h1_bf16 [S], grad_x_route_bf16 [SYM]           │
└──────────────────────────────────────────────────────────────┘
Fallback when the fused kernel is ineligible:
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: standalone SwiGLU backward                           │
│ inputs: grad_h2_bf16, h1_for_bwd_bf16                        │
│ outputs: grad_h1_bf16 [S]                                    │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: BF16 W13 DGRAD + peer scatter                        │
│ inputs: grad_h1_bf16, w13_bf16, bwd_gather_ptrs              │
│ outputs: grad_x_route_bf16 [SYM]                             │
└──────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: BF16 W13 WGRAD                                       │
│ inputs: grad_h1_bf16, x_for_bwd_bf16, expert_counts          │
│ outputs: grad_w13                                            │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
grad_x_route_bf16 [SYM] ── dispatch barrier
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: reduce_from_topk                                     │
│ inputs: grad_x_route_bf16 [SYM], topk_ids                    │
│ outputs: grad_x_bf16 [OUT]                                   │
└──────────────────────────────────────────────────────────────┘

Outputs: grad_x_bf16, grad_scores, grad_w13, grad_w2
```

## NVFP4

### Staged forward

```text
IMPLEMENTED NVFP4 TRAINING FORWARD
-----------------------------------------------------------------------------------------
Inputs: x_bf16, w13_bf16, w2_bf16, topk_ids, topk_scores, slot_id
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: dynamic W13 NVFP4 quant                              │
│ inputs: w13_bf16                                             │
│ outputs: w13_nvfp4 [S; not saved]                            │
└──────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: dynamic W2 NVFP4 quant                               │
│ inputs: w2_bf16                                              │
│ outputs: w2_nvfp4 [S; not saved]                             │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ KERNEL: copy routing                                         │
│ inputs: topk_ids                                             │
│ outputs: routing_ids, routing_token_count                    │
└──────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: q/publish input X                                    │
│ inputs: x_bf16                                               │
│ outputs: x_nvfp4 = q(x_bf16) [SYM]                           │
└──────────────────────────────┬───────────────────────────────┘
                               └─ dispatch barrier ─┐
                                                    ▼
                     ┌──────────────────────────────────────────────────────────────┐
                     │ KERNEL: dist_dispatch_routing                                │
                     │ inputs: routing_ids, routing_token_count                     │
                     │ outputs: expert_counts, recv_by_rank, recv_count,            │
                     │          fwd_gather_ptrs, bwd_gather_ptrs, scatter_ptrs      │
                     └──────────────────────────────┬───────────────────────────────┘
                                                    ▼
                     ┌──────────────────────────────────────────────────────────────┐
                     │ KERNEL: hybrid device forward planner                        │
                     │ inputs: recv_count, recv_by_rank, slot_id                    │
                     │ outputs: need_recompute, recompute_condition,                │
                     │          forward_offsets, activation_offsets, recv_snapshot  │
                     └──────────────────────────────┬───────────────────────────────┘
                                 ┌─────────────────┴──────────────┐
                                 │                                │
                             full-save                        recompute
                                 │                                │
                                 │                  x_saved_bf16 [P]
                                 │                                │
                                 ▼                                ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: staged NVFP4 W13 GEMM + packed-X copy                │
│ inputs: x_nvfp4 [SYM], w13_nvfp4                             │
│         fwd_gather_ptrs, expert_counts                       │
│ outputs: x_gathered_nvfp4 [S], h1_bf16 [P] or [S]            │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ├── x_gathered_nvfp4 [S]
                               │          │ full-save only
                               │          ▼
                               │  ┌───────────────────────────────────────────┐
                               │  │ KERNEL: dq(x_gathered_nvfp4)              │
                               │  │ inputs: x_gathered_nvfp4 [S]              │
                               │  │ outputs: x_q_dq_for_bwd_bf16 [P,M_pad,D]  │
                               │  └───────────────────────────────────────────┘
                               │
                               └── h1_bf16 [P] or [S]
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: q(SwiGLU(h1_bf16)) + W2 GEMM + peer scatter          │
│ inputs: h1_bf16, w2_nvfp4, expert_counts, scatter_ptrs       │
│ outputs: h2_nvfp4 [S], h3_route_bf16 [SYM]                   │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
h3_route_bf16 [SYM] ── combine barrier
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: score × h3_route_bf16 + top-k reduction              │
│ inputs: h3_route_bf16, topk_scores                           │
│ outputs: y_bf16; full-save also writes h3_saved_bf16 [P]     │
└──────────────────────────────────────────────────────────────┘

Full-save: [x_q_dq_for_bwd_bf16 [M_pad,D],
            h1_bf16 [M_pad,2F], h3_saved_bf16 [T,K,D]]
Recompute: [x_saved_bf16 [T,D]]

Fixed state retains every named metadata tensor:
expert_counts, fwd_gather_ptrs, bwd_gather_ptrs, scatter_ptrs,
recv_count, recv_snapshot, need_recompute, recompute_condition,
forward_offsets, activation_offsets, slot and postprocess state.
Native BF16 weights and topk_ids/scores are also retained.

x_gathered_nvfp4, h2_nvfp4, published x_nvfp4, w13_nvfp4, and
w2_nvfp4 are transient. Saved x_q_dq_for_bwd_bf16 and h1_bf16
retain the M_pad routed-row layout on current main.
```

### Backward

```text
IMPLEMENTED NVFP4 TRAINING BACKWARD
-----------------------------------------------------------------------------------------
Same grad_y_bf16 and saved NVFP4-forward state
Native w13_bf16, w2_bf16
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: dynamic W13 NVFP4 quant, unless already prepared     │
│ inputs: w13_bf16                                             │
│ outputs: w13_nvfp4 [S]                                       │
└──────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: dynamic W2 NVFP4 quant, unless already prepared      │
│ inputs: w2_bf16                                              │
│ outputs: w2_nvfp4 [S]                                        │
└──────────────────────────────────────────────────────────────┘
Packed operands coexist; BF16 weight dequant is deferred.

┌──────────────────────────────────────────────────────────────┐
│ KERNEL: device backward planner                              │
│ inputs: need_recompute, recompute_condition,                 │
│         forward_offsets, activation_offsets, recv_count,     │
│         recv_snapshot, slot_id                               │
│ outputs: backward_offsets, backward_activation_offsets       │
└──────────────────────────────┬───────────────────────────────┘
             ┌───────────┴────────────┐
             │                        │
         full-save                 recompute
             │                        │
x_q_dq_for_bwd_bf16 [P,M_pad,D]  x_saved_bf16 [P,T,D]
h1_bf16 [P,M_pad,2F]              │
h3_saved_bf16 [P]                 ▼
             │       ┌───────────────────────────────────────────────┐
             │       │ KERNEL: conditional load saved X              │
             │       │ inputs: x_saved_bf16 [P]                      │
             │       │ outputs: local_x_bf16                         │
             │       └───────────────────────┬───────────────────────┘
             │                               ▼
             │       ┌────────────────────────────────────────────────┐
             │       │ KERNEL: q/publish saved X                      │
             │       │ inputs: local_x_bf16                           │
             │       │ outputs: x_nvfp4 [SYM]                         │
             │       └───────────────────────┬────────────────────────┘
             │                               │ dispatch barrier
             │                               ▼
             │  ┌──────────────────────────────────────────────────────┐
             │  │ KERNEL: NVFP4 W13 reconstruction                     │
             │  │ inputs: x_nvfp4, fwd_gather_ptrs,                    │
             │  │         expert_counts, w13_nvfp4                     │
             │  │ outputs: x_gathered_recomputed_nvfp4 [S],            │
             │  │          h1_recomputed_bf16 [S]                      │
             │  └──────────────────────────┬───────────────────────────┘
             └─────────────────────────────┤

                           │
                           ▼
select h1_for_bwd_bf16
                           │
recompute only:
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: dq(x_gathered_recomputed_nvfp4)                      │
│ inputs: x_gathered_recomputed_nvfp4 [S]                      │
│ outputs: x_q_dq_recomputed_bf16 [S]                          │
└──────────────────────────────┬───────────────────────────────┘
select x_q_dq_for_bwd_bf16:
  full-save [P] or recompute [S]
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: q(SwiGLU) + W2 GEMM + peer scatter                   │
│ inputs: h1_for_bwd_bf16, w2_nvfp4, expert_counts,            │
│         scatter_ptrs                                         │
│ outputs: h2_nvfp4 = q(SwiGLU(h1_for_bwd_bf16)) [S],          │
│          h3_candidate_bf16 [SYM]                             │
└──────────────────────────────┬───────────────────────────────┘
                               ├───────────────────────────────┐
                               │                               │
                               ▼                               │
┌──────────────────────────────────────────────────────────────┐ │
│ KERNEL: dq(h2_nvfp4)                                         │ │
│ inputs: h2_nvfp4 [S]                                         │ │
│ outputs: h2_q_dq_bf16 [S]                                    │ │
└──────────────────────────────────────────────────────────────┘ │
                                                               │ adapter combine barrier
                                                               ▼
                                               h3_candidate peer-visible
                           │
                           ▼
select h3_for_bwd_bf16 = h3_saved or h3_candidate

┌──────────────────────────────────────────────────────────────┐
│ KERNEL: shared BF16 score/reduction backward                 │
│ inputs: grad_y_bf16, topk_scores, h3_for_bwd_bf16            │
│ outputs: grad_h3_route_bf16 [SYM], grad_scores [OUT]         │
└──────────────────────────────┬───────────────────────────────┘
              │ combine barrier
              ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: dq(w2_nvfp4)                                         │
│ inputs: w2_nvfp4                                             │
│ outputs: w2_q_dq_bf16 [S]                                    │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: shared BF16 W2 DGRAD + peer gather                   │
│ inputs: grad_h3_route_bf16, scatter_ptrs, expert_counts,     │
│         w2_q_dq_bf16                                         │
│ outputs: grad_h3_gathered_bf16 [S], grad_h2_bf16 [S]         │
└──────────────────────────────────────────────────────────────┘
              │ w2_q_dq_bf16 released
              ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: shared BF16 W2 WGRAD                                 │
│ inputs: grad_h3_gathered_bf16, h2_q_dq_bf16, expert_counts   │
│ outputs: grad_w2                                             │
└──────────────────────────────┬───────────────────────────────┘
              ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: dq(w13_nvfp4)                                        │
│ inputs: w13_nvfp4                                            │
│ outputs: w13_q_dq_bf16 [S]                                   │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: shared SwiGLU backward + BF16 W13 DGRAD, if eligible │
│ inputs: grad_h2_bf16, h1_for_bwd_bf16, w13_q_dq_bf16,        │
│         bwd_gather_ptrs                                      │
│ outputs: grad_h1_bf16 [S], grad_x_route_bf16 [SYM]           │
└──────────────────────────────────────────────────────────────┘
Fallback when the fused kernel is ineligible:
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: shared standalone SwiGLU backward                    │
│ inputs: grad_h2_bf16, h1_for_bwd_bf16                        │
│ outputs: grad_h1_bf16 [S]                                    │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: shared BF16 W13 DGRAD + peer scatter                 │
│ inputs: grad_h1_bf16, w13_q_dq_bf16, bwd_gather_ptrs         │
│ outputs: grad_x_route_bf16 [SYM]                             │
└──────────────────────────────────────────────────────────────┘
              │ w13_q_dq_bf16 released
              ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: shared BF16 W13 WGRAD                                │
│ inputs: grad_h1_bf16, x_q_dq_for_bwd_bf16, expert_counts     │
│ outputs: grad_w13                                            │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
grad_x_route_bf16 [SYM] ── dispatch barrier
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: shared BF16 reduce_from_topk                         │
│ inputs: grad_x_route_bf16 [SYM], topk_ids                    │
│ outputs: grad_x_bf16 [OUT]                                   │
└──────────────────────────────────────────────────────────────┘

Outputs: grad_x_bf16, grad_scores, grad_w13, grad_w2; slot rewound
```

### Mega forward

```text
IMPLEMENTED NVFP4 MEGA TRAINING FORWARD
-----------------------------------------------------------------------------------------
Inputs: x_bf16, w13_bf16, w2_bf16, topk_ids, topk_scores, slot_id
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: dynamic W13 NVFP4 quant                              │
│ inputs: w13_bf16                                             │
│ outputs: w13_nvfp4 [S; not saved]                            │
└──────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: dynamic W2 NVFP4 quant                               │
│ inputs: w2_bf16                                              │
│ outputs: w2_nvfp4 [S; not saved]                             │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ KERNEL: copy routing                                         │
│ inputs: topk_ids                                             │
│ outputs: routing_ids, routing_token_count                    │
└──────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: q/publish input X                                    │
│ inputs: x_bf16                                               │
│ outputs: x_nvfp4 = q(x_bf16) [SYM]                           │
└──────────────────────────────┬───────────────────────────────┘
                               └─ dispatch barrier ─┐
                                                    ▼
                     ┌──────────────────────────────────────────────────────────────┐
                     │ KERNEL: dist_dispatch_routing                                │
                     │ inputs: routing_ids, routing_token_count                     │
                     │ outputs: expert_counts, recv_by_rank, recv_count,            │
                     │          fwd_gather_ptrs, bwd_gather_ptrs, scatter_ptrs      │
                     └──────────────────────────────┬───────────────────────────────┘
                                                    ▼
                     ┌──────────────────────────────────────────────────────────────┐
                     │ KERNEL: hybrid device forward planner                        │
                     │ inputs: recv_count, recv_by_rank, slot_id                    │
                     │ outputs: need_recompute, recompute_condition,                │
                     │          forward_offsets, activation_offsets, recv_snapshot  │
                     └──────────────────────────────┬───────────────────────────────┘
                                 ┌─────────────────┴──────────────┐
                                 │                                │
                             full-save                        recompute
                                 │                                │
                                 │                  x_saved_bf16 [P]
                                 │                                │
                                 ▼                                ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: Mega NVFP4 W13 + q(SwiGLU) + W2 + peer scatter       │
│ fused: NVFP4 W13 GEMM + q(SwiGLU) + NVFP4 W2 GEMM + scatter  │
│ inputs: x_nvfp4 [SYM], w13_nvfp4, w2_nvfp4,                  │
│         fwd_gather_ptrs, scatter_ptrs, expert_counts         │
│ HBM handoff: W13 writes h1_bf16; q(SwiGLU) reads h1_bf16     │
│              and writes h2_nvfp4; W2 reloads h2_nvfp4        │
│ outputs: h1_bf16 [P or S; HBM], h2_nvfp4 [S; HBM],           │
│          h3_route_bf16 [SYM]                                 │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ├── h1_bf16 [P or S; HBM]
                               ├── h2_nvfp4 [S; HBM]
                               └── h3_route_bf16 [SYM]
                                            │
                                            │
                                            │
                                            │
                                            │
                                            │ full-save only
                                            ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: peer gather/copy packed X                            │
│ inputs: x_nvfp4 [SYM], fwd_gather_ptrs, recv_count           │
│ outputs: x_gathered_nvfp4 [S]                                │
└──────────────────────────────┬───────────────────────────────┘
                               │ dispatch-lifetime barrier
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: dq(x_gathered_nvfp4)                                 │
│ inputs: x_gathered_nvfp4 [S]                                 │
│ outputs: x_q_dq_for_bwd_bf16 [P,M_pad,D]                     │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
h3_route_bf16 [SYM] ── combine barrier
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│ KERNEL: score × h3_route_bf16 + top-k reduction              │
│ inputs: h3_route_bf16, topk_scores                           │
│ outputs: y_bf16; full-save also writes h3_saved_bf16 [P]     │
└──────────────────────────────────────────────────────────────┘

Full-save: [x_q_dq_for_bwd_bf16 [M_pad,D],
            h1_bf16 [M_pad,2F], h3_saved_bf16 [T,K,D]]
Recompute: [x_saved_bf16 [T,D]]

Fixed state retains every named metadata tensor:
expert_counts, fwd_gather_ptrs, bwd_gather_ptrs, scatter_ptrs,
recv_count, recv_snapshot, need_recompute, recompute_condition,
forward_offsets, activation_offsets, slot and postprocess state.
Native BF16 weights and topk_ids/scores are also retained.

x_gathered_nvfp4, h2_nvfp4, published x_nvfp4, w13_nvfp4, and
w2_nvfp4 are transient. Saved x_q_dq_for_bwd_bf16 and h1_bf16
retain the M_pad routed-row layout on current main.
```

At the shared BF16 gradient-math boundary, NVFP4 supplies these reconstructed
operands:

```text
BF16:   x_for_bwd_bf16, h2_bf16, w13_bf16, w2_bf16
NVFP4:  x_q_dq_for_bwd_bf16,
        h2_q_dq_bf16,
        w13_q_dq_bf16,
        w2_q_dq_bf16

w2_q_dq_bf16 and w13_q_dq_bf16 are materialized just in time for their
respective DGRAD kernels and never coexist.
```
