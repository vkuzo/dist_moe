# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Persistent compilation cache used by the standalone CuTe kernels."""

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from torch._vendor.quack import cache as _quack_cache
from torch._vendor.quack.cache.jit import _compute_source_fingerprint


_T = TypeVar("_T")
_DIST_MOE_SOURCE_DIR = Path(__file__).resolve().parents[1]

# Quack fingerprints its own sources and any registered downstream source
# trees. Register all package Python sources so changing compile-time kernel
# helpers cannot reuse a stale object file. The fingerprint is process-global
# and may already have been computed by another PyTorch CuTeDSL operator, so
# invalidate the memoized value when this package adds its tree.
if _DIST_MOE_SOURCE_DIR not in _quack_cache.EXTRA_SOURCE_DIRS:
    _quack_cache.EXTRA_SOURCE_DIRS.append(_DIST_MOE_SOURCE_DIR)
    _compute_source_fingerprint.cache_clear()

jit_cache = _quack_cache.jit_cache


def tvm_ffi_tensor_abi_signature(args: tuple) -> tuple:
    """Return the fixed tensor constraints in a TVM-FFI callable schema.

    CuTe tensors marked shape-dynamic can share one exported artifact across
    extents. Other tensors retain fixed extents in the exported TVM-FFI ABI,
    so those extents must participate in both the process-local and persistent
    cache keys.

    Args:
        args: Positional arguments passed to ``cute.compile``.

    Returns:
        A hashable description of every nested CuTe tensor's dtype, rank,
        dynamic-shape mask, and fixed dimensions.
    """
    tensors: list[tuple] = []

    def visit(value, path: tuple[int, ...]) -> None:
        dynamic_shapes_mask = getattr(value, "dynamic_shapes_mask", None)
        shape = getattr(value, "shape", None)
        element_type = getattr(value, "element_type", None)
        if (
            isinstance(dynamic_shapes_mask, tuple)
            and isinstance(shape, tuple)
            and element_type is not None
            and len(dynamic_shapes_mask) == len(shape)
        ):
            mask = tuple(int(dynamic) for dynamic in dynamic_shapes_mask)
            fixed_shape = tuple(
                None if dynamic else int(dim)
                for dim, dynamic in zip(shape, mask, strict=True)
            )
            tensors.append((path, str(element_type), mask, fixed_shape))
            return
        if isinstance(value, (tuple, list)):
            for index, nested in enumerate(value):
                visit(nested, (*path, index))

    for index, arg in enumerate(args):
        visit(arg, (index,))
    return tuple(tensors)


def compile_with_persistent_cache(
    namespace: str,
    cache_key: tuple,
    compile_fn: Callable[[], _T],
) -> _T:
    """Compile or reload a TVM-FFI CuTe launchable.

    Args:
        namespace: Stable identifier for the caller's cache-key domain.
        cache_key: Pickleable compile-time signature for the launchable.
        compile_fn: Zero-argument callback that performs ``cute.compile`` on
            a cache miss.

    Returns:
        The newly compiled or persistently cached launchable.
    """

    @jit_cache
    def _compile(namespace: str, cache_key: tuple) -> _T:
        return compile_fn()

    return _compile(namespace, cache_key)


__all__ = [
    "compile_with_persistent_cache",
    "jit_cache",
    "tvm_ffi_tensor_abi_signature",
]
