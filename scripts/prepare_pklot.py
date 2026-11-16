"""Prepare the exact archived PKLot split from an extracted official download."""
import argparse
from pathlib import Path
import shutil
import yaml

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--images', type=Path, required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
root = repo / 'data/pklot'
available = {}
for image in args.images.rglob('*.jpg'):
    lot = next((p.lower() for p in image.parts if p.lower() in {'pucpr', 'ufpr04', 'ufpr05'}), None)
    name = image.name if image.name.startswith(('pucpr_', 'ufpr04_', 'ufpr05_')) else f'{lot}_{image.name}'
    available[name] = image
splits = {}
for split, folder in [('train', 'train'), ('val', 'val'), ('test', 'demo')]:
    names = (repo / f'data/splits/pklot_{split}.txt').read_text().splitlines()
    missing = [name for name in names if name not in available]
    if missing:
        raise SystemExit(f'Missing {len(missing)} {split} images, including {missing[:3]}')
    splits[split] = names
for split, folder in [('train', 'train'), ('val', 'val'), ('test', 'demo')]:
    dest = root / 'images' / folder
    dest.mkdir(parents=True, exist_ok=True)
    for name in splits[split]:
        if available[name].resolve() != (dest / name).resolve():
            shutil.copy2(available[name], dest / name)
    print(split, len(splits[split]))
config = {'path': str(root), 'train': 'images/train', 'val': 'images/val', 'test': 'images/demo',
          'names': {0: 'space-empty', 1: 'space-occupied'}}
(repo / 'data/pklot.yaml').write_text(yaml.safe_dump(config))
print('Wrote data/pklot.yaml')
