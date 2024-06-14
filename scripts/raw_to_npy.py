import json
from pathlib import Path
from enum import Enum
from typing import Optional, Tuple
import tyro
import numpy as np

class DTYPES(Enum):
    UINT8 = np.uint8
    UINT16 = np.uint16
    UINT32 = np.uint32
    UINT64 = np.uint64
    INT8 = np.int8
    INT16 = np.int16
    INT32 = np.int32
    INT64 = np.int64
    FLOAT32 = np.float32
    FLOAT64 = np.float64

def main(
    input: Path, 
    resolution: Tuple[int, int, int],
    dtype: Optional[DTYPES] = DTYPES.UINT8,
    output: Optional[Path] = None
):
    assert input.is_file(), f'Input file {input} does not exist'
    dtype = dtype.value
    data = np.fromfile(input, dtype=dtype)
    print(f'Loaded {data.size} elements of type {dtype}')
    vol = data.reshape([resolution[i] for i in [2,1,0]])
    vol = vol.swapaxes(0,2)
    if output is None:
        output = input.with_suffix('.npy')
    np.save(output, vol)
    # TODO: downsample. Realistically we don't need more than 256x256x256
    # TODO: perhaps also do npz compression

if __name__=='__main__':
    tyro.cli(main)