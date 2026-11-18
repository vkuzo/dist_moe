# NVFP4 Changes

## Human note

This fork adds the humans& nvfp4 bwd + bf16 bwd ([link](https://humansand.ai/blog/nvfp4-rl#putting-it-all-together-final-recipe)) to dist-moe.

Changes from original repo, at a high level:
1. add a nvfp4 reference with real quants and real nvfp4 gemms, with bitwise matching actual dist-moe kernels in the fwd and tight tolerance for the bwd (this enables safe hill climbing)
2. add q-dq in the right spots to hook up the pre-existing nvfp4 forward with the pre-existing bf16 bwd, and fix all planner, cuda graphs, etc wiring
3. API updates to enable nvfp4 in a training context, and pass either bf16 or nvfp4 weights to nvfp4 training pipeline
4. four-over-six support (https://github.com/mit-han-lab/fouroversix)
5. comprehesive testing + benchmarks
6. torchtitan RL integration (not landed, and not in this repo) showed small RL runs working as expected

At a high level the kernel changes here are:
A. add fast dequant kernels (to implement the basic recipe), these are standalone
B. add four-over-six support to all existing kernels that need it (no new kernels for this feature)

## Agent note below

This fork adds the following NVFP4 capabilities:

- Replaced the approximate NVFP4 reference and its loose tolerances with a
  real-NVFP4 reference. Forward is bitwise equivalent in matched single-GPU
  coverage, while backward uses very tight tolerances because the grouped CuTe
  kernels and the reference's `torch.mm` operations use different accumulation
  and reduction schedules.
- Hooked up the existing NVFP4 forward kernels to the existing BF16 backward by
  (i) adding the required activation and weight dequantization kernels and (ii)
  implementing the training glue: saved/recomputed activations, scratch
  planning, autograd state, distributed synchronization, FakeTensor/tracing,
  and CUDA graphs.
- Added API support to pass BF16 weights to NVFP4 forward and have them
  quantized dynamically; previously, NVFP4 forward required pre-quantized
  weights.
- Added API support to pass prepared NVFP4 weights to an NVFP4
  training-configured pipeline, enabling external TorchTitan/FSDP integration
  to manage and reuse packed weight buffers.
- Added optional in-kernel Four-over-Six adaptive scaling via
  `use_four_over_six`, with no additional kernel launches. It is supported by
  the source-activation quantizer `_QuantizeNvfp4PerTokenRow`, W13/W2 weight
  quantizer `_triton_quantize_blockscaled_row1d_cublas_blocked`, staged fused
  SwiGLU/H2 quantization kernel
  `dist_blockscaled_grouped_gemm_fprop_swiglu_fwd_combine`, and Mega fused
  forward kernel `chunked_mega_blockscaled_grouped_gemm_fprop_swiglu_fwd`;
  dequantization and GEMM arithmetic are unchanged.
- Added comprehensive testing for staged and Mega pipelines, full-save and
  recompute, distributed execution, autograd, FakeTensor/`make_fx`, and CUDA
  graph capture/replay.
- Added matched BF16, NVFP4 staged, and NVFP4 Mega benchmark and profiling
  support.
- Added detailed execution documentation covering kernel boundaries, saved
  tensors, HBM intermediates, and backward lifetimes.

## Four-GPU Qwen3-30B-A3B Performance

The following measurements use four NVIDIA B200 GPUs with expert parallelism
of four, `D=2048`, `F=768`, 128 global experts, and top-k 8. `T` is the local
token count on each rank. Values are maximum-rank CUDA-graph replay latency,
reported as the median of five windows with 20 iterations per window. A
positive delta means that NVFP4 is slower than BF16.

The training comparison measures one forward and backward using reusable
prepared weights. Refreshing both NVFP4 weight operands from BF16 is outside
the iteration and measured 0.5840 ms separately. The NVFP4 backward continues
to use BF16 GEMMs.

(human note - training path has not been benchmarked or optimized yet)

| Local `T` | BF16 forward + backward | NVFP4 staged forward + BF16 backward | NVFP4 delta |
| ---: | ---: | ---: | ---: |
| 16 | 0.4592 ms | 0.6512 ms | +41.8% |
| 32 | 0.4605 ms | 0.6606 ms | +43.5% |
| 64 | 0.4701 ms | 0.6683 ms | +42.2% |
| 128 | 0.4785 ms | 0.6822 ms | +42.6% |
| 256 | 0.4937 ms | 0.6917 ms | +40.1% |
| 512 | 0.5401 ms | 0.7044 ms | +30.4% |
| 1,024 | 0.6388 ms | 0.9070 ms | +42.0% |
| 2,048 | 0.9422 ms | 1.3227 ms | +40.4% |
| 4,096 | 1.5073 ms | 2.1677 ms | +43.8% |
| 8,192 | 2.6966 ms | 3.8183 ms | +41.6% |
| 16,384 | 5.1165 ms | 7.0572 ms | +37.9% |
| 32,768 | 10.1535 ms | 13.8651 ms | +36.6% |

The rollout comparison measures forward only. NVFP4 weight conversion is
excluded: W13 uses the inference-only fused-SwiGLU interleaved layout, W2 is
also prepared ahead of time, and both prepared operands are reused.

| Local `T` | BF16 forward | NVFP4 interleaved Mega forward | NVFP4 delta |
| ---: | ---: | ---: | ---: |
| 16 | 0.1240 ms | 0.0898 ms | -27.6% |
| 32 | 0.1271 ms | 0.0919 ms | -27.7% |
| 64 | 0.1277 ms | 0.0988 ms | -22.6% |
| 128 | 0.1302 ms | 0.1020 ms | -21.7% |
| 256 | 0.1476 ms | 0.1105 ms | -25.1% |
| 512 | 0.1725 ms | 0.1328 ms | -23.0% |
| 1,024 | 0.2069 ms | 0.1846 ms | -10.8% |
| 2,048 | 0.3261 ms | 0.2817 ms | -13.6% |
| 4,096 | 0.5226 ms | 0.4701 ms | -10.0% |
| 8,192 | 0.9036 ms | 0.8483 ms | -6.1% |
| 16,384 | 1.7119 ms | 1.6080 ms | -6.1% |
| 32,768 | 3.3591 ms | 3.5274 ms | +5.0% |

## Training Peak Memory

For staged training at `T=4096` with the same model shape, one DistMoE context,
one activation slot spanning the full layer stack, dynamic BF16-to-NVFP4
weight preparation, and CUDA graphs, the measured peak allocation was:

| MoE layers | BF16 peak | NVFP4 peak | NVFP4 regression |
| ---: | ---: | ---: | ---: |
| 2 | 2444.796 MiB | 2543.807 MiB | +4.05% |
| 4 | 4301.296 MiB | 4456.338 MiB | +3.60% |
| 8 | 8014.296 MiB | 8277.338 MiB | +3.28% |

The regression approaches approximately 3% as the number of resident MoE
layers grows. NVFP4 saves 29.485 MiB in the context-wide scratch arena, but
that is a one-time saving. Its forward kernels pad each local expert to a
128-row multiple, increasing the saved BF16 X and H1 activation footprint by
27.125 MiB per resident layer for this shape. Dynamic packed-weight preparation
also contributes approximately 75.4 MiB of fixed transient storage because
only one layer's W13 and W2 operands are prepared at a time.

Consequently, the fixed scratch saving and weight-preparation transient become
less important for deeper stacks, while the per-layer padding cost remains.
From the measured four-to-eight-layer slopes, the asymptotic regression for
this isolated shape is approximately 2.9%.
