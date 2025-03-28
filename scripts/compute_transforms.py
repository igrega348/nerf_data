import os
import math
import json
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import tyro
from scipy.spatial.transform import Rotation

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        row_list = []
        for col in row:
            row_list.append(round(col, 10))
        matrix_list.append(row_list)
    return matrix_list

def load_xtekct(fn: Path):
    if isinstance(fn, str): fn = Path(fn)
    if fn.is_file():
        pth = fn
    else:
        pth = next(fn.glob('*.xtekct'))
    assert pth.exists()
    txt = pth.read_text()
    lines = txt.split('\n')
    print(f'Loaded {len(lines)} from "{pth}"')
    data = {}

    for line in lines:
        if ('[' in line) and (']' in line):
            current = line.strip('[]')
            data[current] = {}
        elif len(line)<1:
            pass
        else:
            fields = line.split('=')
            key = fields[0]
            value = '='.join(fields[1:])
            try:
                value = float(value)
            except ValueError:
                pass
            data[current][key] = value
            
    return data

def load_from_ang(pth: Path):
    txt = pth.read_text()
    lines = txt.split('\n')
    print(f'Loaded {len(lines)} lines from "{pth}"')
    # skip 1st line and load from 2nd with delimiter ':'
    data = {}
    for line in lines[1:]:
        if len(line)<1: continue
        key, val = line.split(':')
        data[int(key)] = float(val)
    return data

def load_from_ctdata(pth: Path):
    txt = pth.read_text()
    lines = txt.split('\n')
    print(f'Loaded {len(lines)} lines from "{pth}"')
    for i, line in enumerate(lines):
        if 'Index' in line:
            break
        if 'Angle(deg)' in line:
            columns = {'indices': 'Projection', 'angles': 'Angle(deg)'}
            break
    df = pd.read_csv(pth, skiprows=i, delim_whitespace=True)
    indices = df[columns['indices']].values
    # Due to mismatch in formats, loading from _ctdata required negative sign for angle
    angles = -df[columns['angles']].values
    return dict(zip(indices, angles))

        
def load_angles(fn: Path):
    if isinstance(fn, str): fn = Path(fn)
    if fn.is_file():
        pth = fn
        assert pth.exists()
    else:
        files = list(fn.glob('*.ang')) + list(fn.glob('*_ctdata*'))
        assert len(files) == 1
        pth = files[0]
    if 'ang' in pth.suffix:
        return load_from_ang(pth)
    elif 'ctdata' in pth.stem:
        return load_from_ctdata(pth)
    
def main(
        folder: Path, 
        images_folder: str = 'images',
        xtekct_file: Optional[str] = None,
        angles_file: Optional[str] = None,
        output_fname: Optional[str] = 'transforms.json'
):

    if xtekct_file is not None:
        data = load_xtekct(folder / xtekct_file)
    else:
        data = load_xtekct(folder)
    H = data['XTekCT']['DetectorPixelsX']*data['XTekCT']['DetectorPixelSizeX'] / 2
    L = data['XTekCT']['SrcToDetector']
    alpha = 2*np.arctan(H/L) #* 180 / np.pi
    scale_factor = 2 / (data['XTekCT']['VoxelSizeX']*data['XTekCT']['VoxelsX'])
    R = data['XTekCT']['SrcToObject'] * scale_factor
    print(f'alpha: {alpha*180/np.pi}, R: {R}, scale_factor: {scale_factor}')

    f = data['XTekCT']['DetectorPixelsX'] / 2 / np.tan(alpha/2)
    out_data = {
        'camera_angle_x': alpha,
        'w': data['XTekCT']['DetectorPixelsX'],
        'h': data['XTekCT']['DetectorPixelsY'],
        'cx': data['XTekCT']['DetectorPixelsX'] / 2,
        'cy': data['XTekCT']['DetectorPixelsY'] / 2,
        'fl_x': f,
        'fl_y': f,
        'frames': []
    }

    if angles_file is not None:
        angular_data = load_angles(folder/angles_file)
    else:
        angular_data = load_angles(folder)

    def m4(m: np.ndarray) -> np.ndarray:
        out = np.eye(4)
        out[:3, :3] = m
        return out

    for fn in (folder/images_folder).glob('*.png'):
        proj_num = int(fn.stem.split('_')[-1])
        theta = angular_data[proj_num]    

        cam_matrix = np.eye(4)
        
        th_rad = - np.pi * theta / 180 + np.pi/2 # 0 deg when x-axis pointing left
        pos = R * np.array([np.cos(th_rad), np.sin(th_rad), 0])
        phi = np.arctan2(pos[1], pos[0]) + math.radians(90)

        # Blender way
        cam_matrix[:3, 3] = pos
        cam_matrix = cam_matrix@m4(Rotation.from_rotvec(np.pi/2 * np.array([1,0,0])).as_matrix()) # rotate 90 degrees around x
        cam_matrix = cam_matrix@m4(Rotation.from_rotvec(phi * np.array([0,1,0])).as_matrix())
        # Could do the rotations in one go
        # cam_matrix = m4(Rotation.from_euler('XY', [np.pi/2, phi]).as_matrix())@cam_matrix
        # cam_matrix[:3, 3] = pos
        # or
        # cam_matrix = m4(Rotation.from_euler('xz', [np.pi/2, phi]).as_matrix())@cam_matrix
        # cam_matrix = cam_matrix@m4(Rotation.from_euler('XZ', [np.pi/2, phi]).as_matrix())
        # cam_matrix[:3, 3] = pos

        
        frame_data = {
            'file_path': fn.relative_to(folder).as_posix(),
            'transform_matrix': listify_matrix(cam_matrix)
        }
        out_data['frames'].append(frame_data)

    (folder / output_fname).write_text(json.dumps(out_data, indent=2))
    print(f'Saved {(folder / output_fname).as_posix()} with {len(out_data["frames"])} frames')

if __name__ == '__main__':
    tyro.cli(main)
