"""Train the detector on this road, then measure how well it sees it.

A COCO model finds nothing in a frame from `camera.py` - the cars it learned
are photographs. So the detector is fine-tuned on frames from this simulator,
labelled by the renderer's index pass, which is the same thing a real fleet
does with its own footage and its own labels.

  python3 src/train_detector.py --data data/perception/kl/data.yaml --epochs 40

What comes out is a real detector doing real detection on real pixels. The
driver still never sees a label: it sees what this model reports, misses and
all. The report at the end is what the model actually scores on held-out
frames, so the quality of the driver's eyesight is a measured number rather
than an assumption.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_BASE = "yolo26n.pt"


def train(
    data: Path,
    epochs: int,
    base: str = DEFAULT_BASE,
    imgsz: int = 640,
    batch: int = 16,
    project: Path = Path("checkpoints/perception"),
    name: str = "road",
    device: str | None = None,
) -> dict:
    from ultralytics import YOLO

    # Ultralytics joins a relative `project` onto its own configured runs
    # directory, which puts the weights somewhere outside this repository.
    # Resolving it first keeps a run beside the checkpoints it belongs with.
    project = Path(project).resolve()
    model = YOLO(base)
    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(project),
        name=name,
        exist_ok=True,
        device=device,
        pretrained=True,
        verbose=True,
    )
    metrics = model.val(data=str(data), imgsz=imgsz, device=device, verbose=False)
    report = {
        "base": base,
        "data": str(data),
        "epochs": epochs,
        "imgsz": imgsz,
        "weights": str(Path(project) / name / "weights" / "best.pt"),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": {
            model.names[int(c)]: float(metrics.box.maps[i]) for i, c in enumerate(metrics.box.ap_class_index)
        }
        if len(metrics.box.ap_class_index)
        else {},
    }
    out = Path(project) / name / "report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/perception/kl/data.yaml"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="road")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    report = train(
        args.data,
        args.epochs,
        base=args.base,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
    )
    print(f"[ok] {report['weights']} mAP50={report['map50']:.3f} mAP50-95={report['map50_95']:.3f}")
    for name, score in sorted(report["per_class"].items()):
        print(f"     {name:16s} {score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
