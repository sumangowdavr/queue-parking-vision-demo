"""Convert the archived CrowdHuman subset without redistributing source images."""
import argparse
import json
from pathlib import Path
import zipfile
from PIL import Image
import yaml

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
root = repo / 'data/crowdhuman'

def box(cls, b, width, height):
    x, y, w, h = b
    x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x+w), min(height, y+h)
    if x1-x0 < 2 or y1-y0 < 2:
        return None
    return f'{cls} {(x0+x1)/2/width:.6f} {(y0+y1)/2/height:.6f} {(x1-x0)/width:.6f} {(y1-y0)/height:.6f}'

for split, archive in [('train', 'CrowdHuman_train01.zip'), ('val', 'CrowdHuman_val.zip')]:
    names = (repo / f'data/splits/crowdhuman_{split}.txt').read_text().splitlines()
    wanted = {Path(n).stem for n in names}
    rows = {}
    with (args.source / f'annotation_{split}.odgt').open() as annotations:
        for line in annotations:
            row = json.loads(line)
            if row['ID'] in wanted:
                rows[row['ID']] = row
    assert wanted == rows.keys(), 'Missing selected annotations'
    images = root / 'images' / split
    labels = root / 'labels' / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.source / archive) as source:
        members = {Path(n).name: n for n in source.namelist() if n.endswith('.jpg')}
        for name in names:
            image = images / name
            image.write_bytes(source.read(members[name]))
            with Image.open(image) as im:
                width, height = im.size
            lines = []
            for b in rows[Path(name).stem]['gtboxes']:
                if b.get('tag') != 'person':
                    continue
                if not b.get('extra', {}).get('ignore', 0):
                    lines.append(box(0, b['vbox'], width, height))
                if not b.get('head_attr', {}).get('ignore', 0) and 'hbox' in b:
                    lines.append(box(1, b['hbox'], width, height))
            (labels / Path(name).with_suffix('.txt')).write_text('\n'.join(line for line in lines if line))
    print(split, len(names))
(repo / 'data/crowdhuman.yaml').write_text(yaml.safe_dump({
    'path': str(root), 'train': 'images/train', 'val': 'images/val', 'names': {0: 'person', 1: 'head'}}))
