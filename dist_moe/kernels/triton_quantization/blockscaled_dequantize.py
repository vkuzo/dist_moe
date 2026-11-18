# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Triton transfer and dequantization kernels for NVFP4 backward operands."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .formats import (
    CUBLAS_BLOCKED_ENTRIES_PER_ATOM,
    CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE,
    CUBLAS_BLOCKED_ROW_LANES,
    CUBLAS_BLOCKED_ROWS_PER_ATOM,
    CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM,
)

_BLOCK_SIZE = 256
_ACTIVATION_DEQUANT_BLOCK_Q_SIZE = 2048
_ACTIVATION_DEQUANT_NUM_WARPS = 8
_WEIGHT_DEQUANT_BLOCK_Q_SIZE = 1024
_WEIGHT_DEQUANT_NUM_WARPS = 4
_BF16_ROW_COPY_BLOCK_WORDS = 1024
_BF16_ROW_COPY_PROGRAMS_PER_EXPERT = 64
_TL_CUBLAS_BLOCKED_ENTRIES_PER_ATOM = tl.constexpr(CUBLAS_BLOCKED_ENTRIES_PER_ATOM)
_TL_CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE = tl.constexpr(
    CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE
)
_TL_CUBLAS_BLOCKED_ROW_LANES = tl.constexpr(CUBLAS_BLOCKED_ROW_LANES)
_TL_CUBLAS_BLOCKED_ROWS_PER_ATOM = tl.constexpr(CUBLAS_BLOCKED_ROWS_PER_ATOM)
_TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM = tl.constexpr(
    CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
)


@triton.jit
def _decode_fp4_e2m1(code):
    """Decode one sign-magnitude E2M1 nibble into FP32."""
    magnitude_code = code & 0x7
    magnitude = tl.where(
        magnitude_code <= 4,
        magnitude_code.to(tl.float32) * 0.5,
        tl.where(
            magnitude_code == 5,
            3.0,
            tl.where(magnitude_code == 6, 4.0, 6.0),
        ),
    )
    return tl.where((code & 0x8) != 0, -magnitude, magnitude)


@triton.jit
def _decode_scaled_nvfp4x2_to_bf16x2(packed, scale, global_scale_inv):
    """Decode and scale one packed FP4 pair into one BF16x2 word."""
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b8 q, scale_byte;
            .reg .b16 scale_pair, lo, hi;
            .reg .b32 value_bf16x2, scale_bf16x2;
            .reg .f32 lo_f32, hi_f32;

            cvt.u8.u32 q, $1;
            cvt.u8.u32 scale_byte, $2;
            mov.b16 scale_pair, {scale_byte, scale_byte};
            cvt.rn.bf16x2.e2m1x2 value_bf16x2, q;
            cvt.rn.bf16x2.e4m3x2 scale_bf16x2, scale_pair;
            mul.rn.bf16x2 value_bf16x2, value_bf16x2, scale_bf16x2;
            mov.b32 {lo, hi}, value_bf16x2;
            cvt.f32.bf16 lo_f32, lo;
            cvt.f32.bf16 hi_f32, hi;
            mul.rn.f32 lo_f32, lo_f32, $3;
            mul.rn.f32 hi_f32, hi_f32, $3;
            cvt.rn.bf16x2.f32 $0, hi_f32, lo_f32;
        }
        """,
        constraints="=r,r,r,f",
        args=[
            packed.to(tl.uint32),
            scale.to(tl.uint32),
            global_scale_inv,
        ],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _cublas_blocked_scale_offset(row, col, n_col_blocks):
    """Return the byte offset for one logical CUBLAS-blocked scale."""
    row_atom = row // _TL_CUBLAS_BLOCKED_ROWS_PER_ATOM
    row_in_atom = row - row_atom * _TL_CUBLAS_BLOCKED_ROWS_PER_ATOM
    col_atom = col // _TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
    col_in_atom = col - col_atom * _TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM

    atom_id = row_atom * n_col_blocks + col_atom
    atom_base = atom_id * _TL_CUBLAS_BLOCKED_ENTRIES_PER_ATOM
    row_lane = row_in_atom % _TL_CUBLAS_BLOCKED_ROW_LANES
    row_band = row_in_atom // _TL_CUBLAS_BLOCKED_ROW_LANES
    lane_base = row_lane * _TL_CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE
    return (
        atom_base
        + lane_base
        + row_band * _TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
        + col_in_atom
    )


@triton.jit
def _dequantize_nvfp4_weight_kernel(
    qdata_ptr,
    scale_ptr,
    global_scale_inv_ptr,
    output_ptr,
    global_scale_inv_stride,
    N_COLS: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    GROUP_SCALE_ELEMENTS: tl.constexpr,
    N_COL_BLOCKS: tl.constexpr,
    BLOCK_Q_SIZE: tl.constexpr,
):
    """Decode grouped CUBLAS-blocked NVFP4 weights into contiguous BF16."""
    row = tl.program_id(0)
    row_i64 = row.to(tl.int64)
    group = row // ROWS_PER_GROUP
    row_in_group = row - group * ROWS_PER_GROUP
    global_scale_inv = tl.load(
        global_scale_inv_ptr + group.to(tl.int64) * global_scale_inv_stride,
    ).to(tl.float32)

    qdata = qdata_ptr.to(tl.pointer_type(tl.uint8))
    scales = scale_ptr.to(tl.pointer_type(tl.uint8))
    output = output_ptr.to(tl.pointer_type(tl.uint32))
    q_cols: tl.constexpr = N_COLS // 2
    scale_cols: tl.constexpr = triton.cdiv(N_COLS, 16)
    num_q_blocks: tl.constexpr = triton.cdiv(q_cols, BLOCK_Q_SIZE)
    q_lanes = tl.arange(0, BLOCK_Q_SIZE)
    scale_lanes = tl.arange(0, BLOCK_Q_SIZE // 8)
    for block_idx in tl.static_range(0, num_q_blocks):
        q_cols_in_block = block_idx * BLOCK_Q_SIZE + q_lanes
        valid_q = q_cols_in_block < q_cols
        packed = tl.load(
            qdata + row_i64 * q_cols + q_cols_in_block.to(tl.int64),
            mask=valid_q,
            other=0,
        )
        scale_cols_in_block = block_idx * (BLOCK_Q_SIZE // 8) + scale_lanes
        inner_scale = tl.load(
            scales
            + group.to(tl.int64) * GROUP_SCALE_ELEMENTS
            + _cublas_blocked_scale_offset(
                row_in_group,
                scale_cols_in_block,
                N_COL_BLOCKS,
            ).to(tl.int64),
            mask=scale_cols_in_block < scale_cols,
            other=0,
        )
        scale_per_q = tl.broadcast_to(
            inner_scale[:, None],
            (BLOCK_Q_SIZE // 8, 8),
        ).reshape((BLOCK_Q_SIZE,))
        decoded = _decode_scaled_nvfp4x2_to_bf16x2(
            packed,
            scale_per_q,
            global_scale_inv,
        )
        tl.store(
            output + row_i64 * q_cols + q_cols_in_block.to(tl.int64),
            decoded,
            mask=valid_q,
        )


@triton.jit
def _dequantize_nvfp4_gather_kernel(
    gather_ptrs,
    num_rows_ptr,
    output_buffer_ptr,
    output_offset_ptr,
    condition_ptr,
    n_cols,
    EXECUTE_WHEN_ZERO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    condition = tl.load(condition_ptr, eviction_policy="evict_last")
    should_execute = condition == 0 if EXECUTE_WHEN_ZERO else condition != 0
    if not should_execute:
        return

    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    num_rows = tl.load(num_rows_ptr, eviction_policy="evict_last")
    valid_col = cols < n_cols
    valid_row = row < num_rows
    row_address = tl.load(gather_ptrs + row)
    valid_source = valid_row & (row_address != 0)

    packed_ptr = row_address.to(tl.pointer_type(tl.uint8))
    packed = tl.load(
        packed_ptr + cols // 2,
        mask=valid_col & valid_source,
        other=0,
    )
    code = tl.where((cols & 1) == 0, packed & 0xF, packed >> 4)
    values = _decode_fp4_e2m1(code)

    q_cols = n_cols // 2
    scale_cols = n_cols // 16
    scale_ptr = row_address.to(tl.pointer_type(tl.float8e4nv))
    inner_scale = tl.load(
        scale_ptr + q_cols + cols // 16,
        mask=valid_col & valid_source,
        other=0.0,
    ).to(tl.float32)
    global_scale_inv_ptr = (row_address + q_cols + scale_cols).to(
        tl.pointer_type(tl.float32)
    )
    global_scale_inv = tl.load(
        global_scale_inv_ptr,
        mask=valid_source,
        other=0.0,
    ).to(tl.float32)

    output_offset = tl.load(
        output_offset_ptr,
        eviction_policy="evict_last",
    ).to(tl.int64)
    output = (output_buffer_ptr + output_offset).to(tl.pointer_type(tl.bfloat16))
    tl.store(
        output + row.to(tl.int64) * n_cols + cols.to(tl.int64),
        values * inner_scale * global_scale_inv,
        mask=valid_col & valid_row,
    )


@triton.jit
def _copy_gathered_nvfp4_to_activation_buffer_kernel(
    gather_ptrs,
    num_rows_ptr,
    activation_buffer_ptr,
    qdata_offset_ptr,
    scale_offset_ptr,
    condition_ptr,
    n_cols,
    EXECUTE_WHEN_ZERO: tl.constexpr,
    ROW_SCALE_STORAGE_MULTIPLE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy peer-addressed packed rows into one planner-owned row bundle."""
    condition = tl.load(condition_ptr, eviction_policy="evict_last")
    should_execute = condition == 0 if EXECUTE_WHEN_ZERO else condition != 0
    if not should_execute:
        return

    row = tl.program_id(0)
    byte_cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    num_rows = tl.load(num_rows_ptr, eviction_policy="evict_last").to(tl.int64)
    row_i64 = row.to(tl.int64)
    valid_row = row_i64 < num_rows
    row_address = tl.load(gather_ptrs + row, mask=valid_row, other=0)
    valid_source = valid_row & (row_address != 0)
    source = row_address.to(tl.pointer_type(tl.uint8))

    q_cols = n_cols // 2
    valid_q = byte_cols < q_cols
    packed = tl.load(source + byte_cols, mask=valid_q & valid_source, other=0)
    qdata_offset = tl.load(qdata_offset_ptr, eviction_policy="evict_last").to(tl.int64)
    qdata = activation_buffer_ptr + qdata_offset
    tl.store(
        qdata + row_i64 * q_cols + byte_cols.to(tl.int64),
        packed,
        mask=valid_q & valid_row,
    )

    scale_cols = n_cols // 16
    valid_scale = byte_cols < scale_cols
    scale = tl.load(
        source + q_cols + byte_cols,
        mask=valid_scale & valid_source,
        other=0,
    )
    scale_offset = tl.load(scale_offset_ptr, eviction_policy="evict_last").to(tl.int64)
    scales = activation_buffer_ptr + scale_offset
    n_col_blocks = (
        scale_cols + _TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM - 1
    ) // _TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
    blocked_scale_offset = _cublas_blocked_scale_offset(
        row,
        byte_cols,
        n_col_blocks,
    ).to(tl.int64)
    tl.store(
        scales + blocked_scale_offset,
        scale,
        mask=valid_scale & valid_row,
    )

    source_global_scale_inv = tl.load(
        (source + q_cols + scale_cols).to(tl.pointer_type(tl.float32))
        + tl.zeros((BLOCK_SIZE,), tl.int64),
        mask=(byte_cols == 0) & valid_source,
        other=0.0,
    )
    global_scale_offset = (
        scale_offset + num_rows * ROW_SCALE_STORAGE_MULTIPLE * scale_cols
    )
    global_scale_inv = (activation_buffer_ptr + global_scale_offset).to(
        tl.pointer_type(tl.float32)
    )
    tl.store(
        global_scale_inv + row_i64 + tl.zeros((BLOCK_SIZE,), tl.int64),
        source_global_scale_inv,
        mask=(byte_cols == 0) & valid_row,
    )


@triton.jit
def _dequantize_nvfp4_activation_buffer_kernel(
    activation_buffer_ptr,
    qdata_offset_ptr,
    scale_offset_ptr,
    num_rows_ptr,
    output_offset_ptr,
    condition_ptr,
    N_COLS: tl.constexpr,
    EXECUTE_WHEN_ZERO: tl.constexpr,
    EXECUTE_UNCONDITIONALLY: tl.constexpr,
    ROW_SCALE_STORAGE_MULTIPLE: tl.constexpr,
    BLOCK_Q_SIZE: tl.constexpr,
):
    """Decode one planner-owned row bundle into planner-owned BF16 storage."""
    if not EXECUTE_UNCONDITIONALLY:
        condition = tl.load(condition_ptr, eviction_policy="evict_last")
        should_execute = condition == 0 if EXECUTE_WHEN_ZERO else condition != 0
        if not should_execute:
            return

    row = tl.program_id(0)
    num_rows = tl.load(num_rows_ptr, eviction_policy="evict_last").to(tl.int64)
    row_i64 = row.to(tl.int64)
    if row_i64 >= num_rows:
        return

    qdata_offset = tl.load(qdata_offset_ptr, eviction_policy="evict_last").to(tl.int64)
    scale_offset = tl.load(scale_offset_ptr, eviction_policy="evict_last").to(tl.int64)
    qdata = (activation_buffer_ptr + qdata_offset).to(tl.pointer_type(tl.uint8))
    scales = (activation_buffer_ptr + scale_offset).to(tl.pointer_type(tl.uint8))

    scale_cols: tl.constexpr = N_COLS // 16
    n_col_blocks = (scale_cols + _TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM - 1) // (
        _TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
    )
    global_scale_offset = (
        scale_offset + num_rows * ROW_SCALE_STORAGE_MULTIPLE * scale_cols
    )
    global_scale_inv = tl.load(
        (activation_buffer_ptr + global_scale_offset).to(tl.pointer_type(tl.float32))
        + row_i64,
    ).to(tl.float32)

    output_offset = tl.load(
        output_offset_ptr,
        eviction_policy="evict_last",
    ).to(tl.int64)
    output = (activation_buffer_ptr + output_offset).to(tl.pointer_type(tl.uint32))
    q_cols: tl.constexpr = N_COLS // 2
    num_q_blocks: tl.constexpr = triton.cdiv(q_cols, BLOCK_Q_SIZE)
    q_lanes = tl.arange(0, BLOCK_Q_SIZE)
    scale_lanes = tl.arange(0, BLOCK_Q_SIZE // 8)
    for block_idx in tl.static_range(0, num_q_blocks):
        q_cols_in_block = block_idx * BLOCK_Q_SIZE + q_lanes
        valid_q = q_cols_in_block < q_cols
        packed = tl.load(
            qdata + row_i64 * q_cols + q_cols_in_block.to(tl.int64),
            mask=valid_q,
            other=0,
        )
        scale_cols_in_block = block_idx * (BLOCK_Q_SIZE // 8) + scale_lanes
        inner_scale = tl.load(
            scales
            + _cublas_blocked_scale_offset(
                row,
                scale_cols_in_block,
                n_col_blocks,
            ).to(tl.int64),
            mask=scale_cols_in_block < scale_cols,
            other=0,
        )
        scale_per_q = tl.broadcast_to(
            inner_scale[:, None],
            (BLOCK_Q_SIZE // 8, 8),
        ).reshape((BLOCK_Q_SIZE,))
        decoded = _decode_scaled_nvfp4x2_to_bf16x2(
            packed,
            scale_per_q,
            global_scale_inv,
        )
        tl.store(
            output + row_i64 * q_cols + q_cols_in_block.to(tl.int64),
            decoded,
            mask=valid_q,
        )


@triton.jit
def _routed_row_mapping(
    row,
    padded_counts_ptr,
    unpadded_counts_ptr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
    EXPAND: tl.constexpr,
):
    """Map one row between padded and sentinel-free expert segments."""
    experts = tl.arange(0, BLOCK_EXPERTS)
    expert_mask = experts < NUM_LOCAL_EXPERTS
    padded_counts = tl.load(
        padded_counts_ptr + experts,
        mask=expert_mask,
        other=0,
    ).to(tl.int64)
    unpadded_counts = tl.load(
        unpadded_counts_ptr + experts,
        mask=expert_mask,
        other=0,
    ).to(tl.int64)
    lookup_counts = padded_counts if EXPAND else unpadded_counts
    lookup_ends = tl.cumsum(lookup_counts, axis=0)
    expert = tl.sum(
        tl.where(expert_mask, row >= lookup_ends, False).to(tl.int32),
        axis=0,
    )
    expert_before = experts < expert
    lookup_start = tl.sum(
        tl.where(expert_before, lookup_counts, 0),
        axis=0,
    )
    padded_start = tl.sum(
        tl.where(expert_before, padded_counts, 0),
        axis=0,
    )
    unpadded_start = tl.sum(
        tl.where(expert_before, unpadded_counts, 0),
        axis=0,
    )
    within_expert = row - lookup_start
    actual_count = tl.sum(
        tl.where(experts == expert, unpadded_counts, 0),
        axis=0,
    )
    if EXPAND:
        source_row = unpadded_start + within_expert
        destination_row = row
        valid_value = within_expert < actual_count
    else:
        source_row = padded_start + within_expert
        destination_row = row
        valid_value = True
    return source_row, destination_row, valid_value


@triton.jit
def _copy_routed_bf16_rows_kernel(
    activation_buffer_ptr,
    source_offset_ptr,
    destination_offset_ptr,
    padded_counts_ptr,
    unpadded_counts_ptr,
    num_source_rows_ptr,
    num_destination_rows_ptr,
    condition_ptr,
    N_COLS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
    BLOCK_WORDS: tl.constexpr,
    EXPAND: tl.constexpr,
    EXECUTE_WHEN_ZERO: tl.constexpr,
):
    """Copy BF16 rows between padded and sentinel-free expert layouts."""
    condition = tl.load(condition_ptr, eviction_policy="evict_last")
    should_execute = condition == 0 if EXECUTE_WHEN_ZERO else condition != 0
    if not should_execute:
        return

    num_source_rows = tl.load(
        num_source_rows_ptr,
        eviction_policy="evict_last",
    ).to(tl.int64)
    num_destination_rows = tl.load(
        num_destination_rows_ptr,
        eviction_policy="evict_last",
    ).to(tl.int64)
    if num_source_rows == num_destination_rows:
        return
    experts = tl.arange(0, BLOCK_EXPERTS)
    expert_mask = experts < NUM_LOCAL_EXPERTS
    padded_counts = tl.load(
        padded_counts_ptr + experts,
        mask=expert_mask,
        other=0,
    ).to(tl.int64)
    unpadded_counts = tl.load(
        unpadded_counts_ptr + experts,
        mask=expert_mask,
        other=0,
    )
    expert = tl.program_id(0)
    expert_before = experts < expert
    padded_start = tl.sum(
        tl.where(expert_before, padded_counts, 0),
        axis=0,
    )
    unpadded_start = tl.sum(
        tl.where(expert_before, unpadded_counts, 0),
        axis=0,
    )
    actual_count = tl.sum(
        tl.where(experts == expert, unpadded_counts, 0),
        axis=0,
    )
    padded_count = tl.sum(
        tl.where(experts == expert, padded_counts, 0),
        axis=0,
    )
    rows_to_copy = padded_count if EXPAND else actual_count
    source_start = unpadded_start if EXPAND else padded_start
    destination_start = padded_start if EXPAND else unpadded_start

    source_offset = tl.load(
        source_offset_ptr,
        eviction_policy="evict_last",
    ).to(tl.int64)
    destination_offset = tl.load(
        destination_offset_ptr,
        eviction_policy="evict_last",
    ).to(tl.int64)
    source = (activation_buffer_ptr + source_offset).to(tl.pointer_type(tl.uint32))
    destination = (activation_buffer_ptr + destination_offset).to(
        tl.pointer_type(tl.uint32)
    )
    words_per_row: tl.constexpr = N_COLS // 2
    word_lanes = tl.arange(0, BLOCK_WORDS)
    row_in_expert = tl.program_id(1).to(tl.int64)
    while row_in_expert < rows_to_copy:
        source_row = source_start + row_in_expert
        destination_row = destination_start + row_in_expert
        valid_value = (not EXPAND) or row_in_expert < actual_count
        valid_value &= source_row < num_source_rows
        valid_destination = destination_row < num_destination_rows
        for block_start in tl.static_range(0, words_per_row, BLOCK_WORDS):
            words = block_start + word_lanes
            word_mask = words < words_per_row
            values = tl.load(
                source + source_row * words_per_row + words,
                mask=word_mask & valid_value,
                other=0,
            )
            tl.store(
                destination + destination_row * words_per_row + words,
                values,
                mask=word_mask & valid_destination,
            )
        row_in_expert += tl.num_programs(1)


@triton.jit
def _dequantize_compact_nvfp4_activation_buffer_kernel(
    activation_buffer_ptr,
    qdata_offset_ptr,
    scale_offset_ptr,
    padded_counts_ptr,
    unpadded_counts_ptr,
    padded_num_rows_ptr,
    unpadded_num_rows_ptr,
    output_offset_ptr,
    condition_ptr,
    N_COLS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
    EXECUTE_WHEN_ZERO: tl.constexpr,
    EXECUTE_UNCONDITIONALLY: tl.constexpr,
    ROW_SCALE_STORAGE_MULTIPLE: tl.constexpr,
    BLOCK_Q_SIZE: tl.constexpr,
):
    """Decode padded NVFP4 rows into sentinel-free BF16 expert segments."""
    if not EXECUTE_UNCONDITIONALLY:
        condition = tl.load(condition_ptr, eviction_policy="evict_last")
        should_execute = condition == 0 if EXECUTE_WHEN_ZERO else condition != 0
        if not should_execute:
            return

    output_row = tl.program_id(0).to(tl.int64)
    unpadded_num_rows = tl.load(
        unpadded_num_rows_ptr,
        eviction_policy="evict_last",
    ).to(tl.int64)
    if output_row >= unpadded_num_rows:
        return
    padded_num_rows = tl.load(
        padded_num_rows_ptr,
        eviction_policy="evict_last",
    ).to(tl.int64)
    source_row, _, _ = _routed_row_mapping(
        output_row,
        padded_counts_ptr,
        unpadded_counts_ptr,
        NUM_LOCAL_EXPERTS=NUM_LOCAL_EXPERTS,
        BLOCK_EXPERTS=BLOCK_EXPERTS,
        EXPAND=False,
    )

    qdata_offset = tl.load(qdata_offset_ptr, eviction_policy="evict_last").to(tl.int64)
    scale_offset = tl.load(scale_offset_ptr, eviction_policy="evict_last").to(tl.int64)
    qdata = (activation_buffer_ptr + qdata_offset).to(tl.pointer_type(tl.uint8))
    scales = (activation_buffer_ptr + scale_offset).to(tl.pointer_type(tl.uint8))
    scale_cols: tl.constexpr = N_COLS // 16
    n_col_blocks = (scale_cols + _TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM - 1) // (
        _TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
    )
    global_scale_offset = (
        scale_offset + padded_num_rows * ROW_SCALE_STORAGE_MULTIPLE * scale_cols
    )
    global_scale_inv = tl.load(
        (activation_buffer_ptr + global_scale_offset).to(tl.pointer_type(tl.float32))
        + source_row,
    ).to(tl.float32)

    output_offset = tl.load(
        output_offset_ptr,
        eviction_policy="evict_last",
    ).to(tl.int64)
    output = (activation_buffer_ptr + output_offset).to(tl.pointer_type(tl.uint32))
    q_cols: tl.constexpr = N_COLS // 2
    num_q_blocks: tl.constexpr = triton.cdiv(q_cols, BLOCK_Q_SIZE)
    q_lanes = tl.arange(0, BLOCK_Q_SIZE)
    scale_lanes = tl.arange(0, BLOCK_Q_SIZE // 8)
    for block_idx in tl.static_range(0, num_q_blocks):
        q_cols_in_block = block_idx * BLOCK_Q_SIZE + q_lanes
        valid_q = q_cols_in_block < q_cols
        packed = tl.load(
            qdata + source_row * q_cols + q_cols_in_block.to(tl.int64),
            mask=valid_q,
            other=0,
        )
        scale_cols_in_block = block_idx * (BLOCK_Q_SIZE // 8) + scale_lanes
        inner_scale = tl.load(
            scales
            + _cublas_blocked_scale_offset(
                source_row,
                scale_cols_in_block,
                n_col_blocks,
            ).to(tl.int64),
            mask=scale_cols_in_block < scale_cols,
            other=0,
        )
        scale_per_q = tl.broadcast_to(
            inner_scale[:, None],
            (BLOCK_Q_SIZE // 8, 8),
        ).reshape((BLOCK_Q_SIZE,))
        decoded = _decode_scaled_nvfp4x2_to_bf16x2(
            packed,
            scale_per_q,
            global_scale_inv,
        )
        tl.store(
            output + output_row * q_cols + q_cols_in_block.to(tl.int64),
            decoded,
            mask=valid_q,
        )


def dequantize_nvfp4_weight(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    global_scale_inv: torch.Tensor,
    *,
    rows_per_group: int,
) -> torch.Tensor:
    """Dequantize contiguous grouped NVFP4 weights into contiguous BF16.

    Args:
        qdata: Contiguous rank-two packed E2M1 rows.
        scale: Contiguous grouped CUBLAS-blocked E4M3 scales.
        global_scale_inv: Per-group FP32 inverse outer scales.
        rows_per_group: Logical weight rows belonging to one expert.

    Returns:
        Contiguous BF16 rows with twice the packed column count.
    """
    rows, q_cols = qdata.shape
    n_cols = q_cols * 2
    output = torch.empty((rows, n_cols), dtype=torch.bfloat16, device=qdata.device)
    if rows == 0 or n_cols == 0:
        return output

    scale_cols = triton.cdiv(n_cols, 16)
    n_col_blocks = triton.cdiv(scale_cols, CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM)
    padded_rows = (
        triton.cdiv(rows_per_group, CUBLAS_BLOCKED_ROWS_PER_ATOM)
        * CUBLAS_BLOCKED_ROWS_PER_ATOM
    )
    group_scale_elements = (
        padded_rows * n_col_blocks * CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
    )
    _dequantize_nvfp4_weight_kernel[(rows,)](
        qdata.view(torch.uint8),
        scale,
        global_scale_inv,
        output,
        global_scale_inv.stride(0),
        N_COLS=n_cols,
        ROWS_PER_GROUP=rows_per_group,
        GROUP_SCALE_ELEMENTS=group_scale_elements,
        N_COL_BLOCKS=n_col_blocks,
        BLOCK_Q_SIZE=_WEIGHT_DEQUANT_BLOCK_Q_SIZE,
        num_warps=_WEIGHT_DEQUANT_NUM_WARPS,
    )
    return output


def dequantize_gathered_nvfp4_to_activation_buffer(
    gather_ptrs: torch.Tensor,
    num_rows: torch.Tensor,
    activation_buffer: torch.Tensor,
    activation_offset: torch.Tensor,
    condition: torch.Tensor,
    *,
    n_cols: int,
    execute_when_zero: bool,
) -> None:
    """Dequantize peer-addressed packed rows into planner-owned BF16 storage.

    Args:
        gather_ptrs: Capacity-sized table of packed symmetric-memory row addresses.
        num_rows: Device scalar containing the valid padded-row count.
        activation_buffer: Planner-owned one-dimensional byte storage.
        activation_offset: One-element device byte offset for the destination.
        condition: One-element device predicate.
        n_cols: Logical BF16 feature width.
        execute_when_zero: Whether a zero predicate selects this operation.
    """
    if gather_ptrs.numel() == 0 or n_cols == 0:
        return
    grid = (gather_ptrs.numel(), triton.cdiv(n_cols, _BLOCK_SIZE))
    _dequantize_nvfp4_gather_kernel[grid](
        gather_ptrs,
        num_rows,
        activation_buffer,
        activation_offset,
        condition,
        n_cols,
        EXECUTE_WHEN_ZERO=execute_when_zero,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=4,
    )


def copy_gathered_nvfp4_to_activation_buffer(
    gather_ptrs: torch.Tensor,
    num_rows: torch.Tensor,
    activation_buffer: torch.Tensor,
    qdata_offset: torch.Tensor,
    scale_offset: torch.Tensor,
    condition: torch.Tensor,
    *,
    capacity_rows: int,
    n_cols: int,
    execute_when_zero: bool,
    row_scale_storage_multiple: int,
) -> None:
    """Copy peer-addressed packed NVFP4 rows into planner-owned storage.

    Args:
        gather_ptrs: Capacity-sized table of packed symmetric-memory row addresses.
        num_rows: Device scalar containing the valid padded-row count.
        activation_buffer: Planner-owned one-dimensional byte storage.
        qdata_offset: Device scalar byte offset of the packed E2M1 destination.
        scale_offset: Device scalar byte offset of the blocked E4M3-scale bundle.
        condition: One-element device predicate.
        capacity_rows: Static maximum row count used for graph-stable launch.
        n_cols: Logical BF16 feature width.
        execute_when_zero: Whether a zero predicate selects this operation.
        row_scale_storage_multiple: Physical row-scale padding multiplier.
    """
    if capacity_rows == 0 or n_cols == 0:
        return
    grid = (capacity_rows, triton.cdiv(n_cols // 2, _BLOCK_SIZE))
    _copy_gathered_nvfp4_to_activation_buffer_kernel[grid](
        gather_ptrs,
        num_rows,
        activation_buffer,
        qdata_offset,
        scale_offset,
        condition,
        n_cols,
        EXECUTE_WHEN_ZERO=execute_when_zero,
        ROW_SCALE_STORAGE_MULTIPLE=row_scale_storage_multiple,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=4,
    )


def dequantize_nvfp4_activation_buffer_to_activation_buffer(
    activation_buffer: torch.Tensor,
    qdata_offset: torch.Tensor,
    scale_offset: torch.Tensor,
    num_rows: torch.Tensor,
    output_offset: torch.Tensor,
    condition: torch.Tensor,
    *,
    capacity_rows: int,
    n_cols: int,
    execute_when_zero: bool,
    execute_unconditionally: bool,
    row_scale_storage_multiple: int,
) -> None:
    """Dequantize one offset-backed NVFP4 row bundle into BF16 storage.

    Args:
        activation_buffer: Planner-owned byte-addressed allocation.
        qdata_offset: Device scalar byte offset of packed E2M1 rows.
        scale_offset: Device scalar byte offset of CUBLAS-blocked E4M3 scales.
        num_rows: Device scalar containing the valid padded-row count.
        output_offset: Device scalar byte offset of the BF16 destination.
        condition: Device predicate selecting this operation.
        capacity_rows: Static maximum row count used for graph-stable launch.
        n_cols: Logical BF16 feature width.
        execute_when_zero: Whether a zero predicate selects this operation.
        execute_unconditionally: Whether to ignore the predicate and always decode.
        row_scale_storage_multiple: Physical row-scale padding multiplier.
    """
    if capacity_rows == 0 or n_cols == 0:
        return
    grid = (capacity_rows,)
    _dequantize_nvfp4_activation_buffer_kernel[grid](
        activation_buffer,
        qdata_offset,
        scale_offset,
        num_rows,
        output_offset,
        condition,
        N_COLS=n_cols,
        EXECUTE_WHEN_ZERO=execute_when_zero,
        EXECUTE_UNCONDITIONALLY=execute_unconditionally,
        ROW_SCALE_STORAGE_MULTIPLE=row_scale_storage_multiple,
        BLOCK_Q_SIZE=_ACTIVATION_DEQUANT_BLOCK_Q_SIZE,
        num_warps=_ACTIVATION_DEQUANT_NUM_WARPS,
    )


def copy_routed_bf16_activation_buffer_rows(
    activation_buffer: torch.Tensor,
    source_offset: torch.Tensor,
    destination_offset: torch.Tensor,
    padded_counts: torch.Tensor,
    unpadded_counts: torch.Tensor,
    num_source_rows: torch.Tensor,
    num_destination_rows: torch.Tensor,
    condition: torch.Tensor,
    *,
    destination_capacity_rows: int,
    n_cols: int,
    expand: bool,
    execute_when_zero: bool,
) -> None:
    """Copy BF16 expert rows between padded and sentinel-free layouts.

    Args:
        activation_buffer: Planner-owned byte-addressed allocation.
        source_offset: Source byte-offset device scalar.
        destination_offset: Destination byte-offset device scalar.
        padded_counts: Per-expert counts including sentinels.
        unpadded_counts: Per-expert real-row counts.
        num_source_rows: Source-row count device scalar.
        num_destination_rows: Destination-row count device scalar.
        condition: Device predicate for the copy.
        destination_capacity_rows: Static graph-stable launch capacity.
        n_cols: BF16 values per row.
        expand: Whether compact rows are expanded and padding is zeroed.
        execute_when_zero: Whether a zero predicate executes the copy.
    """
    if destination_capacity_rows == 0 or n_cols == 0:
        return
    _copy_routed_bf16_rows_kernel[
        (
            padded_counts.numel(),
            min(
                destination_capacity_rows,
                _BF16_ROW_COPY_PROGRAMS_PER_EXPERT,
            ),
        )
    ](
        activation_buffer,
        source_offset,
        destination_offset,
        padded_counts,
        unpadded_counts,
        num_source_rows,
        num_destination_rows,
        condition,
        N_COLS=n_cols,
        NUM_LOCAL_EXPERTS=padded_counts.numel(),
        BLOCK_EXPERTS=triton.next_power_of_2(padded_counts.numel()),
        BLOCK_WORDS=_BF16_ROW_COPY_BLOCK_WORDS,
        EXPAND=expand,
        EXECUTE_WHEN_ZERO=execute_when_zero,
        num_warps=4,
    )


def dequantize_compact_nvfp4_activation_buffer_to_activation_buffer(
    activation_buffer: torch.Tensor,
    qdata_offset: torch.Tensor,
    scale_offset: torch.Tensor,
    padded_counts: torch.Tensor,
    unpadded_counts: torch.Tensor,
    padded_num_rows: torch.Tensor,
    unpadded_num_rows: torch.Tensor,
    output_offset: torch.Tensor,
    condition: torch.Tensor,
    *,
    unpadded_capacity_rows: int,
    n_cols: int,
    execute_when_zero: bool,
    execute_unconditionally: bool,
    row_scale_storage_multiple: int,
) -> None:
    """Decode padded NVFP4 rows into sentinel-free BF16 storage.

    Args:
        activation_buffer: Planner-owned byte-addressed allocation.
        qdata_offset: Packed E2M1 source byte-offset device scalar.
        scale_offset: Block-scale source byte-offset device scalar.
        padded_counts: Per-expert counts including sentinels.
        unpadded_counts: Per-expert real-row counts.
        padded_num_rows: Total padded-row device scalar.
        unpadded_num_rows: Total real-row device scalar.
        output_offset: Compact BF16 destination byte-offset device scalar.
        condition: Device predicate for the decode.
        unpadded_capacity_rows: Static graph-stable destination capacity.
        n_cols: Logical values per row.
        execute_when_zero: Whether a zero predicate executes the decode.
        execute_unconditionally: Whether the predicate is ignored.
        row_scale_storage_multiple: Physical blocked-scale row multiplier.
    """
    if unpadded_capacity_rows == 0 or n_cols == 0:
        return
    _dequantize_compact_nvfp4_activation_buffer_kernel[(unpadded_capacity_rows,)](
        activation_buffer,
        qdata_offset,
        scale_offset,
        padded_counts,
        unpadded_counts,
        padded_num_rows,
        unpadded_num_rows,
        output_offset,
        condition,
        N_COLS=n_cols,
        NUM_LOCAL_EXPERTS=padded_counts.numel(),
        BLOCK_EXPERTS=triton.next_power_of_2(padded_counts.numel()),
        EXECUTE_WHEN_ZERO=execute_when_zero,
        EXECUTE_UNCONDITIONALLY=execute_unconditionally,
        ROW_SCALE_STORAGE_MULTIPLE=row_scale_storage_multiple,
        BLOCK_Q_SIZE=_ACTIVATION_DEQUANT_BLOCK_Q_SIZE,
        num_warps=_ACTIVATION_DEQUANT_NUM_WARPS,
    )
