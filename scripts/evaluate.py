"""Evaluate existing checkpoints; keep new results separate from archived metrics."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from ultralytics import YOLO

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--parking-data')
parser.add_argument('--crowd-data')
parser.add_argument('--device', default='cpu')
args = parser.parse_args()
if not args.parking_data and not args.crowd_data:
    parser.error('Supply --parking-data or --crowd-data')
repo = Path(__file__).resolve().parents[1]
results = {}
options = dict(imgsz=640, batch=8, device=args.device, plots=False, project=str(repo/'runs'))
if args.parking_data:
    m = YOLO(repo/'models/pklot_obb_best.pt').val(data=args.parking_data, split='test', name='parking_test', **options)
    results['parking_heldout'] = {'mAP50': float(m.box.map50), 'mAP50-95': float(m.box.map),
        'per_class_mAP50': dict(zip(['space-empty', 'space-occupied'], map(float, m.box.ap50)))}
if args.crowd_data:
    m = YOLO(repo/'models/crowd_y11n_best.pt').val(data=args.crowd_data, name='crowd_test', **options)
    results['crowd_finetuned'] = dict(zip(['mAP50_person', 'mAP50_head'], map(float, m.box.ap50)))
    stock = YOLO('yolo11n.pt').val(data=args.crowd_data, classes=[0], name='crowd_stock', **options)
    results['crowd_stock_coco'] = {'mAP50_person': float(stock.box.ap50[0])}
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
dest = repo/f'evaluation-{stamp}.json'
dest.write_text(json.dumps(results, indent=2)+'\n')
print(dest)
