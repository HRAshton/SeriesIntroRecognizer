from typing import TypeVar

import cupy as cp  # type: ignore

T1 = TypeVar('T1')
T2 = TypeVar('T2')
T3 = TypeVar('T3')

type GpuInt = cp.ndarray[cp.int32]
type GpuFloat = cp.ndarray[cp.float32]
type GpuFloatArray = cp.ndarray[GpuFloat]
type GpuStack[T1, T2, T3] = cp.ndarray[T1, T2, T3]
