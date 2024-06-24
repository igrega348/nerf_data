import json
from pathlib import Path
from enum import Enum
from typing import Optional, Tuple
import tyro
import numpy as np
from scipy.interpolate import interpn
import cv2 as cv

def main(
    input: Path, 
):
    assert input.is_file(), f'Input file {input} does not exist'
    if input.suffix == '.npz':
        data = np.load(input)['vol']
    else:
        data = np.load(input)
    print(f'Loaded {data.size} elements of type {data.dtype}')
    # Show 3 slices, one from each axis. Create sliders to change the slice. Also add a slider to threshold the image
    def on_change_x(val):
        x = cv.getTrackbarPos('x', 'slices')
        y = cv.getTrackbarPos('y', 'slices')
        z = cv.getTrackbarPos('z', 'slices')
        mmin = cv.getTrackbarPos('min', 'slices')
        mmax = cv.getTrackbarPos('max', 'slices')
        _data = np.clip(data, mmin, mmax)
        _data = (_data - mmin) / (mmax - mmin)
        imshow = np.hstack([
            _data[x,:,::-1].T,
            _data[:,y,::-1].T,
            _data[:,::-1,z].T,
        ])
        cv.imshow('slices', imshow)
    cv.namedWindow('slices', cv.WINDOW_GUI_NORMAL)
    cv.createTrackbar('x', 'slices', 0, data.shape[0]-1, on_change_x)
    cv.createTrackbar('y', 'slices', 0, data.shape[1]-1, on_change_x)
    cv.createTrackbar('z', 'slices', 0, data.shape[2]-1, on_change_x)
    cv.createTrackbar('min', 'slices', 0, 255, on_change_x)
    cv.createTrackbar('max', 'slices', 255, 255, on_change_x)
    on_change_x(0)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__=='__main__':
    tyro.cli(main)