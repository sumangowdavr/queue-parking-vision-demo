"""Parking-lot occupancy demo on a held-out PKLot day (never used in training).
Model classes: 0 = space-empty, 1 = space-occupied. Shows counts, % full, an alert rule,
and the ground-truth count from the dataset labels so the error is visible.

Usage:
  python parking_demo.py --model best.pt --lot pucpr --date 2012-10-29 --out parking.mp4
"""
import argparse, glob, os, subprocess
import cv2, numpy as np
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--lot', default='pucpr'); ap.add_argument('--date', default='2012-10-29')
ap.add_argument('--root', default='data/pklot')
ap.add_argument('--imgsz', type=int, default=640); ap.add_argument('--conf', type=float, default=0.35)
ap.add_argument('--full-pct', type=float, default=85); ap.add_argument('--hold', type=int, default=3,
                help='consecutive snapshots (5 min apart) above full-pct before alerting')
ap.add_argument('--sec-per-frame', type=float, default=0.35); ap.add_argument('--device', default='cpu')
ap.add_argument('--title', default=''); ap.add_argument('--model-label', default='')
a = ap.parse_args()

imgs = sorted(glob.glob(f'{a.root}/images/demo/{a.lot}_{a.date}_*.jpg'))
assert imgs, 'no images for that lot/date'
model = YOLO(a.model)
W0 = 1280; fps = 25; rep = max(1, round(a.sec_per_frame * fps))
h0 = None; ff = None
series, gt_series, errs = [], [], []; above = 0; alert = False; alerts = 0

def put(img, txt, xy, sc=0.6, col=(255, 255, 255), th=1):
    cv2.putText(img, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)

for p in imgs:
    img = cv2.imread(p); H, W = img.shape[:2]
    r = model.predict(img, imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0]
    occ = emp = 0
    if getattr(r, 'obb', None) is not None:   # rotated boxes (YOLO-OBB)
        polys, clss = r.obb.xyxyxyxy.cpu().numpy(), r.obb.cls.int().cpu().tolist()
    else:
        polys = [np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) for x0, y0, x1, y1 in r.boxes.xyxy.cpu().numpy()]
        clss = r.boxes.cls.int().cpu().tolist()
    for poly, c in zip(polys, clss):
        col = (60, 60, 230) if c == 1 else (60, 200, 60)
        occ += c == 1; emp += c == 0
        cv2.polylines(img, [np.array(poly, np.int32)], True, col, 2)
    lab = open(p.replace('/images/', '/labels/')[:-4] + '.txt').read().split('\n')
    gt_occ = sum(1 for l in lab if l.startswith('1 ')); gt_tot = sum(1 for l in lab if l.strip())
    tot = occ + emp; pct = 100 * occ / max(gt_tot, 1)
    series.append(pct); gt_series.append(100 * gt_occ / max(gt_tot, 1)); errs.append(abs(occ - gt_occ))
    above = above + 1 if pct >= a.full_pct else 0
    if above >= a.hold and not alert: alert = True; alerts += 1
    if above == 0: alert = False
    s = W0 / W; out = cv2.resize(img, (W0, int(H * s) // 2 * 2)); oh = out.shape[0]
    if ff is None:
        ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W0}x{oh}',
                               '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-threads', '2', '-crf', '22', '-pix_fmt', 'yuv420p',
                               '-movflags', '+faststart', a.out], stdin=subprocess.PIPE)
    hhmm = os.path.basename(p)[len(a.lot) + 12:len(a.lot) + 17].replace('_', ':')
    pw, ph, px, py = 390, 232, 16, 16
    pnl = out.copy(); cv2.rectangle(pnl, (px, py), (px + pw, py + ph), (20, 20, 20), -1)
    out = cv2.addWeighted(pnl, 0.75, out, 0.25, 0)
    put(out, a.title or f'Parking lot {a.lot.upper()}  -  {a.date}  {hhmm}', (px + 14, py + 30), 0.62, (255, 255, 255), 2)
    put(out, f'Occupied {occ}  /  Free {emp}', (px + 14, py + 66), 0.85, (255, 255, 255), 2)
    put(out, f'{pct:4.0f}% full', (px + 14, py + 102), 0.95, (0, 200, 255) if alert else (90, 230, 90), 2)
    put(out, f'Labels say: {gt_occ} occupied of {gt_tot}', (px + 170, py + 100), 0.5, (200, 200, 200))
    put(out, f'Rule: alert if >= {a.full_pct:g}% full for {a.hold} snapshots ({a.hold * 5} min)', (px + 14, py + 130), 0.47, (200, 200, 200))
    sx, sy, sw, sh = px + 14, py + 142, pw - 28, 78
    cv2.rectangle(out, (sx, sy), (sx + sw, sy + sh), (80, 80, 80), 1)
    yl = sy + sh - int(a.full_pct * sh / 100); cv2.line(out, (sx, yl), (sx + sw, yl), (0, 140, 255), 1)
    n = len(imgs)
    for ser, col, th in ((gt_series, (150, 150, 150), 1), (series, (90, 230, 90), 2)):
        pts = [(sx + int(i * sw / max(n - 1, 1)), sy + sh - int(v * sh / 100)) for i, v in enumerate(ser)]
        if len(pts) > 1: cv2.polylines(out, [np.array(pts, np.int32)], False, col, th)
    put(out, 'model', (sx + sw - 110, sy + 14), 0.4, (90, 230, 90)); put(out, 'labels', (sx + sw - 55, sy + 14), 0.4, (170, 170, 170))
    if alert:
        cv2.rectangle(out, (0, oh - 54), (W0, oh), (0, 0, 200), -1)
        put(out, f'ALERT: lot {pct:.0f}% full - send arriving cars to the overflow lot', (20, oh - 18), 0.8, (255, 255, 255), 2)
    if a.model_label: put(out, a.model_label, (W0 - 14 - 9 * len(a.model_label), 30), 0.5, (230, 230, 230))
    for _ in range(rep): ff.stdin.write(out.tobytes())

ff.stdin.close(); ff.wait()
print(f'{len(imgs)} snapshots  mean abs error {np.mean(errs):.2f} spaces/snapshot  '
      f'max {max(errs)}  alerts {alerts}')
