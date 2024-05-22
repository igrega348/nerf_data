import json
import os
import sys
from pathlib import Path
import tyro


def main(folder: Path):
    """Combine transform files from folder into a single file.

    Args:
        folder (Path): path to folder containing transform files.
            transform files are named transforms_*.json. 
    """
    assert folder.is_dir()
    transforms = None
    for fn in folder.glob('transforms_*.json'):
        timestamp = int(fn.stem.split('_')[-1])
        t = timestamp*0.1
        print(fn)
        d = json.loads(fn.read_text())
        for f in d['frames']:
            assert 'time' in f
            f['time'] = round(t,2)
        if transforms is None:
            transforms = d
        else:
            transforms['frames'].extend(d['frames'])
    out_fname = folder/'transforms.json'
    print(f'Writing to {out_fname}')
    out_fname.write_text(json.dumps(transforms, indent=2))

if __name__=='__main__':
    tyro.cli(main)
