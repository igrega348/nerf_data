# %%
from typing import Optional, Literal, Any
import cv2 as cv
import numpy as np
from pathlib import Path
from rich.progress import track
import tyro
from enum import Enum
# %%
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

m_thresh_min = None
m_thresh_max = None

def main(
        input: Path,
        output_dir: Path,
        thresh_min: Optional[float] = None,
        thresh_max: Optional[float] = None,
        dtype: Optional[DTYPES] = DTYPES.UINT8
):
    assert input.exists()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {int(fn.stem.split('_')[-1]):fn for fn in input.glob('*.tif')}
    nums = list(files.keys())
    print(f'Found {len(files)} tiff files in {input}')

    global m_thresh_min, m_thresh_max
    if thresh_min is not None:
        m_thresh_min = thresh_min
    if thresh_max is not None:
        m_thresh_max = thresh_max

    dtype = dtype.value

    def on_trackbar_thresh_min(val):
        global m_thresh_min
        m_thresh_min = val
        update_image()

    def on_trackbar_thresh_max(val):
        global m_thresh_max
        m_thresh_max = val
        update_image()

    def on_trackbar_rot(val):
        global img
        try:
            img = cv.imread(str(files[val]), cv.IMREAD_UNCHANGED)
        except KeyError:
            pass
        update_image()

    def update_image():
        global m_thresh_min, m_thresh_max, img
        img_clip = threshold_image_colormap(img, m_thresh_min, m_thresh_max, DTYPES.UINT8.value)
        cv.imshow('image', img_clip)

    def threshold_image_colormap(
            img: np.ndarray,
            thresh_min: float, thresh_max: float,
            dtype: Optional[Any] = None
    ) -> np.ndarray:
        if dtype is None:
            dtype = img.dtype
        max_val = np.iinfo(dtype).max
        img_clip = img.astype(np.float64)
        vals_below = img < thresh_min
        vals_above = img > thresh_max
        img_clip = np.clip(img, thresh_min, thresh_max)
        # rescale between min and max
        img_clip = (img_clip - thresh_min) / (thresh_max - thresh_min) * max_val
        img_clip = img_clip.astype(dtype)
        # apply colormap
        img_clip = cv.applyColorMap(img_clip, cv.COLORMAP_JET)
        g = 0.5*max_val
        grey = np.array([g, g, g], dtype=dtype)
        img_clip[vals_below] = grey
        img_clip[vals_above] = grey
        return img_clip

    def threshold_one_image(
            img: np.ndarray, 
            thresh_min: float, thresh_max: float, 
            dtype: Optional[Any] = None
    ) -> np.ndarray:
        if dtype is None:
            dtype = img.dtype
        max_val = np.iinfo(dtype).max
        img_clip = img.astype(np.float64)
        img_clip = np.clip(img, thresh_min, thresh_max)
        # rescale between min and max
        img_clip = (img_clip - thresh_min) / (thresh_max - thresh_min) * max_val
        img_clip = img_clip.astype(dtype)
        return img_clip

    if thresh_max is None or thresh_min is None:
        img = cv.imread(str(files[1]), cv.IMREAD_UNCHANGED)
        max_val = np.iinfo(img.dtype).max
        print(f'Input dtype {img.dtype}, max value {max_val}')
        if m_thresh_min is None:
            m_thresh_min = 0
        if m_thresh_max is None:
            m_thresh_max = max_val
        cv.namedWindow('image', cv.WINDOW_NORMAL)
        cv.createTrackbar('threshold_min', 'image', 0, max_val, on_trackbar_thresh_min)
        cv.createTrackbar('threshold_max', 'image', 0, max_val, on_trackbar_thresh_max)
        cv.createTrackbar('rotation', 'image', min(nums), max(nums), on_trackbar_rot)
        cv.waitKey(50)
        k = cv.waitKey(0)
        cv.destroyAllWindows()
        print(f'Chosen thresholds: {m_thresh_min}, {m_thresh_max}')

    tif_files = list(input.glob('*.tif'))
    for fn in track(tif_files, description='Thresholding tiff and saving as png'):
        img = cv.imread(str(fn), cv.IMREAD_UNCHANGED)
        img = threshold_one_image(img, m_thresh_min, m_thresh_max, dtype)
        cv.imwrite(str((output_dir/fn.stem).with_suffix('.png')), img)

if __name__ == '__main__':
    tyro.cli(main)