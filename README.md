# People, queues and parking occupancy

A small computer-vision project I built on November 15–16, 2026 to explore queue and parking monitoring. It turns detections into two useful signals: a queue that stays long, and a parking lot that is nearly full. This is my independent research demo.

[Watch the project](https://sumangowdavr.github.io/projects/queue-parking-vision.html) · [Queue recording](media/queue-people.mp4) · [Parking recording](media/parking-ufpr05.mp4) · [Glare failure](media/parking-glare.mp4)

The queue monitor uses YOLO11n and ByteTrack, a polygon around the queue, a dwell filter and a sustained-count alert. The parking model uses rotated boxes to distinguish empty and occupied spaces, with the dataset's occupied-space count displayed beside the prediction. Both trained checkpoints are in `models/`.

## Recorded results

These are saved results from the November 15–16 run, preserved in [`results/recorded_results.json`](results/recorded_results.json). They are subset evaluations, not official benchmark leaderboard scores.

| Evaluation | Saved result | Scope |
| --- | --- | --- |
| Parking detection | mAP50 **0.941**, mAP50–95 **0.804** | 460 snapshots across three held-out camera-days |
| UFPR04 occupied-count error | **0.05** spaces/snapshot mean absolute error; max 2 | 156 snapshots, 2012-12-23 |
| UFPR05 occupied-count error | **1.14** spaces/snapshot; max 5 | 143 snapshots, 2013-04-10 |
| PUCPR occupied-count error | **11.47** spaces/snapshot; max 79 | 161 snapshots, 2012-10-29; strong afternoon glare |
| Person detection, fine-tuned / stock COCO | mAP50 **0.669 / 0.602** | Same 300-image CrowdHuman validation subset; visible-person boxes |
| Head detection | mAP50 **0.581** | Same subset |
| Queue model + tracker | **168.9 ms/frame**, one alert | Laptop CPU, 960px inference, 475-frame stock clip |

Parking training: 900 images from all three camera views, 150 validation images, 15 epochs at 640px. People training: 1,200 CrowdHuman images, 300 validation images, 15 epochs at 640px. Training used a laptop CPU. Exact image lists and PKLot converted annotations are included under `data/`.

Whole held-out days are excluded from parking training and validation, but the camera views are represented in training. This does not establish generalization to a new camera. Queue timing is measured from visible track entry during the clip; it is zone dwell, not verified service wait time. Stock and fine-tuned people models looked similar on this one queue recording.

## Run it

Python 3.10 and FFmpeg are required. The recorded environment used PyTorch 2.5.1, torchvision 0.20.1, Ultralytics 8.3.40 and OpenCV. Install PyTorch for your machine, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python queue_demo.py --help
python parking_demo.py --help
```

For a video you have permission to use, draw a queue polygon in its original pixel coordinates:

```bash
python queue_demo.py --video /path/to/queue.mp4 --out queue-output.mp4 \
  --model models/crowd_y11n_best.pt --cls 0 \
  --zone '120,740;1920,800;1920,975;120,860' --min-count 6 --hold 4
```

That polygon belongs to the original 1920px Coverr clip; redraw it for your camera. A person must stay in the zone for three seconds before counting. The count is smoothed over one second. The example alert requires six people for four seconds.

Download and extract [PKLot](https://web.inf.ufpr.br/vri/databases/parking-lot-database/), then prepare exactly the saved split using the included converted annotations:

```bash
python scripts/prepare_pklot.py --images /path/to/extracted/PKLot
python parking_demo.py --model models/pklot_obb_best.pt \
  --root data/pklot --lot ufpr05 --date 2013-04-10 --out parking-output.mp4
python scripts/evaluate.py --parking-data data/pklot.yaml
```

The parking denominator comes from the annotated number of spaces, representing a known camera layout. Its alert uses 85% occupancy for three consecutive snapshots. The snapshots are approximately five minutes apart; the code assumes five minutes when labeling the alert interval. Playback is accelerated to 0.36 seconds per snapshot.

To prepare the people subset, obtain `CrowdHuman_train01.zip`, `CrowdHuman_val.zip`, `annotation_train.odgt` and `annotation_val.odgt` from [CrowdHuman](https://www.crowdhuman.org/download.html), put them in one directory, then:

```bash
python scripts/prepare_crowdhuman.py --source /path/to/crowdhuman
python scripts/evaluate.py --crowd-data data/crowdhuman.yaml
```

To repeat the original training configuration after data preparation:

```bash
yolo obb train model=yolo11n-obb.pt data=data/pklot.yaml epochs=15 imgsz=640 batch=8 device=cpu workers=4 seed=0 patience=6 project=runs name=pklot_obb
yolo detect train model=yolo11n.pt data=data/crowdhuman.yaml epochs=15 imgsz=640 batch=8 device=cpu workers=4 seed=0 patience=6 project=runs name=crowd_y11n
```

New evaluation output is written separately from the saved results. GPU, library and hardware changes can affect training and timing.

## What failed

The PUCPR recording exposes a late-afternoon glare and shadow failure that an overall mAP score hides. A later six-epoch brightness-augmentation experiment was tried locally; the published checkpoints and videos here remain the evaluated original models. The repository does not claim that the glare issue is solved.

The original parking render also permits overlapping empty and occupied detections for the same space. Counts are direct detection counts, not one prediction assigned to each fixed stall. The original videos preserve this behavior. Class-agnostic suppression or fixed-stall matching needs a separate evaluation.

The head model detects heads, but its original queue configuration did not provide reliable zone tracking and produced no queue alert. The reported head diagnostic used the default head-to-feet offset of zero; the CLI offers a heuristic offset, but that option was not used in the saved diagnostic. A higher detection score did not establish better queue monitoring.

## Sources and use

See [THIRD_PARTY.md](THIRD_PARTY.md). Original project code is AGPL-3.0, consistent with its Ultralytics dependency. Data and media keep their own terms. CrowdHuman images and image-containing training plots are excluded; its model is provided for non-commercial research and educational demonstrations. This is a recorded prototype.
