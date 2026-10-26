
from pathlib import Path
import yaml

ROOT = Path(r"D:\AI ml project data set\extracted_datasets")

DATASETS = [
    "chilli labelling 3.v2i.yolov11",
    "chilli project 4.v1i.yolov11",
    "chilli ventemole 1",
    "white fly 2",
]

for name in DATASETS:
    folder = ROOT / name
    cfg_path = folder / "data.yaml"

    print("\n" + "=" * 60)
    print("DATASET:", name)

    if not cfg_path.exists():
        print("ERROR: data.yaml missing")
        continue

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]

    print("nc:", cfg.get("nc"))
    print("names:", names)

    for split in ("train", "valid", "test"):
        image_dir = folder / split / "images"
        label_dir = folder / split / "labels"

        images = sum(
            1 for p in image_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            }
        ) if image_dir.exists() else 0

        labels = len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0

        print(f"{split}: images={images}, label_files={labels}")
