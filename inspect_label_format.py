
from pathlib import Path
import yaml

ROOT = Path(r"D:\AI ml project data set\extracted_datasets")

datasets = [
    "chilli labelling 3.v2i.yolov11",
    "chilli project 4.v1i.yolov11",
    "chilli ventemole 1",
    "white fly 2",
]

for name in datasets:
    folder = ROOT / name
    config_path = folder / "data.yaml"

    print("\n" + "=" * 65)
    print("DATASET:", name)

    if not config_path.exists():
        print("Missing data.yaml")
        continue

    config = yaml.safe_load(
        config_path.read_text(encoding="utf-8-sig")
    )
    print("Classes:", config.get("names"))

    label_dir = folder / "train" / "labels"
    if not label_dir.exists():
        print("Training labels folder missing")
        continue

    shown = 0
    for file in sorted(label_dir.glob("*.txt")):
        lines = [
            line.strip()
            for line in file.read_text(
                encoding="utf-8-sig"
            ).splitlines()
            if line.strip()
        ]

        for line in lines:
            values = line.split()
            print("File:", file.name)
            print("Number of values:", len(values))
            print("First 20 values:", values[:20])

            if len(values) == 5:
                print("Possible format: YOLO bounding box")
            elif len(values) >= 7 and len(values) % 2 == 1:
                print("Possible format: YOLO segmentation polygon")
            else:
                print("Format needs further inspection")

            shown += 1
            break

        if shown >= 3:
            break
