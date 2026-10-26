
from pathlib import Path
import yaml

ROOT = Path(r"D:\AI ml project data set\extracted_datasets")

datasets = [
    "chilli labelling 3.v2i.yolov11",
    "chilli project 4.v1i.yolov11",
    "chilli ventemole 1",
    "white fly 2",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for dataset_name in datasets:
    dataset = ROOT / dataset_name
    yaml_path = dataset / "data.yaml"

    print("\n" + "=" * 60)
    print("DATASET:", dataset_name)

    if not yaml_path.exists():
        print("ERROR: data.yaml not found")
        continue

    with yaml_path.open("r", encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)

    names = config.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]

    print("Declared classes:", names)
    print("Number of classes:", config.get("nc"))

    for split in ("train", "valid", "test"):
        label_dir = dataset / split / "labels"
        image_dir = dataset / split / "images"

        if not label_dir.exists():
            print(f"{split}: label folder missing")
            continue

        labels = list(label_dir.glob("*.txt"))
        images = [
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ] if image_dir.exists() else []

        image_stems = {p.stem for p in images}
        bad = []
        empty = []
        checked = 0

        for label_file in labels:
            checked += 1
            matching_image = label_file.stem in image_stems

            if not matching_image:
                bad.append(f"{label_file.name}: matching image missing")
                continue

            lines = label_file.read_text(
                encoding="utf-8-sig"
            ).splitlines()

            if not any(line.strip() for line in lines):
                empty.append(label_file.name)
                continue

            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue

                parts = line.split()

                try:
                    values = [float(x) for x in parts]
                    valid = (
                        len(values) == 5
                        and values[0].is_integer()
                        and 0 <= int(values[0]) < len(names)
                        and all(0 <= x <= 1 for x in values[1:])
                        and values[3] > 0
                        and values[4] > 0
                    )
                except ValueError:
                    valid = False

                if not valid:
                    bad.append(
                        f"{label_file.name}, line {line_number}: "
                        f"{len(parts)} values; {line[:100]}"
                    )

        print(
            f"{split}: images={len(images)}, labels={len(labels)}, "
            f"checked={checked}, empty={len(empty)}, issues={len(bad)}"
        )

        for issue in bad[:3]:
            print("  ISSUE:", issue)

        for name in empty[:3]:
            print("  EMPTY:", name)
