"""Queue monitor demo: YOLO + ByteTrack + a queue zone.
Counts people standing in the zone, time each person has waited, smoothed queue length,
and raises an alert when the line stays at/above a threshold for a set time.

Usage:
  python queue_demo.py --video in.mp4 --out out.mp4 --model yolo11n.pt \
      --zone "120,740;1920,800;1920,975;120,860" --min-count 6 --hold 5
"""
import argparse, collections, time, subprocess
import cv2, numpy as np
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('--video', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--model', default='yolo11n.pt'); ap.add_argument('--cls', type=int, default=0)
ap.add_argument('--zone', required=True, help='x,y;x,y;... in source pixels (feet area)')
ap.add_argument('--min-count', type=int, default=6); ap.add_argument('--hold', type=float, default=5.0)
ap.add_argument('--min-dwell', type=float, default=3.0, help='seconds in zone before someone counts as queuing')
ap.add_argument('--imgsz', type=int, default=960); ap.add_argument('--conf', type=float, default=0.3)
ap.add_argument('--title', default='Queue monitor'); ap.add_argument('--model-label', default='')
ap.add_argument('--head-to-feet', type=float, default=0.0,
                help='0 for person boxes; ~6.5 for head boxes (feet = chin + k * head height)')
ap.add_argument('--out-width', type=int, default=1280); ap.add_argument('--device', default='cpu')
a = ap.parse_args()

zone = np.array([[int(v) for v in p.split(',')] for p in a.zone.split(';')], np.int32)
model = YOLO(a.model)
cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS) or 25
W, H = int(cap.get(3)), int(cap.get(4)); s = a.out_width / W
oh = int(H * s) // 2 * 2
ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{a.out_width}x{oh}',
                       '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-threads', '2', '-crf', '22', '-preset', 'medium',
                       '-pix_fmt', 'yuv420p', '-movflags', '+faststart', a.out], stdin=subprocess.PIPE)

first_seen = {}          # track id -> first frame inside zone
last_seen = {}
served_waits = []        # dwell of people who left the zone after queuing
hist = collections.deque(maxlen=int(fps * 1.0))   # 1 s rolling median of the count
series = []              # smoothed count per frame (for the sparkline)
above_since = None; alert = False; alerts = 0; f = 0; t0 = time.time(); inf_ms = []

def put(img, txt, xy, sc=0.6, col=(255, 255, 255), th=1):
    cv2.putText(img, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)

while True:
    ok, frame = cap.read()
    if not ok: break
    t1 = time.time()
    r = model.track(frame, persist=True, tracker='bytetrack.yaml', classes=[a.cls],
                    imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0]
    inf_ms.append((time.time() - t1) * 1000)
    in_zone_now = set(); boxes = []
    if r.boxes.id is not None:
        for (x0, y0, x1, y1), tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist()):
            # heads: the box ends at the chin, so estimate the feet ~6.5 head-heights lower
            foot = (float((x0 + x1) / 2), float(y1 + a.head_to_feet * (y1 - y0)))
            inside = cv2.pointPolygonTest(zone, foot, False) >= 0
            if inside:
                first_seen.setdefault(tid, f); last_seen[tid] = f; in_zone_now.add(tid)
            boxes.append((x0, y0, x1, y1, tid, inside))
    # people who left the zone (gone > 1 s) after queuing long enough -> a completed wait
    for tid in list(first_seen):
        if tid not in in_zone_now and f - last_seen[tid] > fps:
            d = (last_seen[tid] - first_seen[tid]) / fps
            if d >= a.min_dwell: served_waits.append(d)
            first_seen.pop(tid); last_seen.pop(tid)
    queuing = [t for t in in_zone_now if (f - first_seen[t]) / fps >= a.min_dwell]
    hist.append(len(queuing)); q = int(np.median(hist)); series.append(q)
    if q >= a.min_count:
        above_since = above_since if above_since is not None else f
        if not alert and (f - above_since) / fps >= a.hold: alert = True; alerts += 1
    else:
        above_since = None; alert = False
    waits = [(f - first_seen[t]) / fps for t in queuing]

    # ---- draw ----
    ov = frame.copy(); cv2.fillPoly(ov, [zone], (0, 200, 255) if alert else (0, 180, 0))
    frame = cv2.addWeighted(ov, 0.18, frame, 0.82, 0)
    cv2.polylines(frame, [zone], True, (0, 0, 255) if alert else (0, 220, 0), 3)
    for x0, y0, x1, y1, tid, inside in boxes:
        q_ok = inside and tid in queuing
        col = (60, 220, 60) if q_ok else ((0, 200, 255) if inside else (170, 170, 170))
        cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), col, 3 if inside else 1)
        if inside:
            put(frame, f'#{tid} {(f - first_seen[tid]) / fps:4.1f}s', (int(x0), int(y0) - 8), 0.9, col, 2)
    out = cv2.resize(frame, (a.out_width, oh))
    # panel
    pw, ph = 360, 210; px, py = 16, 16
    pnl = out.copy(); cv2.rectangle(pnl, (px, py), (px + pw, py + ph), (20, 20, 20), -1)
    out = cv2.addWeighted(pnl, 0.72, out, 0.28, 0)
    put(out, a.title, (px + 14, py + 30), 0.7, (255, 255, 255), 2)
    put(out, f'In line now: {q}', (px + 14, py + 66), 0.9, (90, 230, 90) if not alert else (0, 200, 255), 2)
    put(out, f'Longest time in line: {max(waits) if waits else 0:4.1f} s*', (px + 14, py + 96), 0.58)
    put(out, f'Avg wait of people who left: {np.mean(served_waits) if served_waits else 0:4.1f} s', (px + 14, py + 120), 0.58)
    put(out, f'Rule: alert if >= {a.min_count} in line for {a.hold:g} s   *since clip start', (px + 14, py + 146), 0.5, (200, 200, 200))
    # sparkline
    sx, sy, sw, sh = px + 14, py + 160, pw - 28, 40
    cv2.rectangle(out, (sx, sy), (sx + sw, sy + sh), (80, 80, 80), 1)
    if len(series) > 1:
        m = max(max(series), a.min_count + 1); n = len(series)
        pts = [(sx + int(i * sw / max(n - 1, 1)), sy + sh - int(v * sh / m)) for i, v in enumerate(series)]
        cv2.line(out, (sx, sy + sh - int(a.min_count * sh / m)), (sx + sw, sy + sh - int(a.min_count * sh / m)), (0, 140, 255), 1)
        cv2.polylines(out, [np.array(pts, np.int32)], False, (90, 230, 90), 2)
    if alert:
        cv2.rectangle(out, (0, oh - 54), (a.out_width, oh), (0, 0, 200), -1)
        put(out, f'ALERT: line has been {q}+ people for over {a.hold:g} s - open another register', (20, oh - 18), 0.8, (255, 255, 255), 2)
    if a.model_label:
        put(out, a.model_label, (a.out_width - 12 - 9 * len(a.model_label), 30), 0.5, (230, 230, 230))
    ff.stdin.write(out.tobytes()); f += 1

ff.stdin.close(); ff.wait()
print(f'frames {f}  avg model+tracker {np.mean(inf_ms):.1f} ms/frame  alerts {alerts}  '
      f'completed waits {len(served_waits)}  peak smoothed count {max(series) if series else 0}')
