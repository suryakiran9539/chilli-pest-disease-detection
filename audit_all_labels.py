
from pathlib import Path
from collections import Counter
import yaml

ROOT = Path(r"D:\AI ml project data set\extracted_datasets")

DATASETS = [
    "chilli labelling 3.v2i.yolov11",
    "chilli project 4.v1i.yolov11",
    "chilli ventemole 1",
    "white fly 2",
]

for dataset_name in DATASETS:
    folder = ROOT / dataset_name
    yaml_path = folder / "data.yaml"

    print("\n" + "=" * 65)
    print("DATASET:", dataset_name)

    if not yaml_path.exists():
        print("ERROR: data.yaml missing")
        continue

    config = yaml.safe_load(
        yaml_path.read_text(encoding="utf-8-sig")
    )
    names = config.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]

    print("Classes:", names)

    for split in ("train", "valid", "test"):
        label_dir = folder / split / "labels"
        if not label_dir.exists():
            print(split, "label folder missing")
            continue

        counts = Counter()
        examples = {}

        for file in label_dir.glob("*.txt"):
            text = file.read_text(encoding="utf-8-sig")
            for line_no, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue

                parts = line.split()
                try:
                    values = [float(v) for v in parts]
                except ValueError:
                    counts["non_numeric"] += 1
                    examples.setdefault("non_numeric", file.name)
                    continue

                if len(values) == 5:
                    class_id = values[0]
                    coords = values[1:]
                    if (
                        class_id.is_integer()
                        and 0 <= int(class_id) < len(names)
                        and all(0 <= v <= 1 for v in coords)
                        and coords[2] > 0
                        and coords[3] > 0
                    ):
                        counts["bbox"] += 1
                    else:
                        counts["invalid_5_values"] += 1
                        examples.setdefault("invalid_5_values", file.name)
                elif (
                    len(values) >= 7
                    and len(values) % 2 == 1
                    and values[0].is_integer()
                    and 0 <= int(values[0]) < len(names)
                    and all(0 <= v <= 1 for v in values[1:])
                ):
                    counts["possible_polygon"] += 1
                    examples.setdefault("possible_polygon", file.name)
                else:
                    counts["other_or_invalid"] += 1
                    examples.setdefault("other_or_invalid", file.name)

        print(f"\n{split.upper()} annotation lines:")
        for category in (
            "bbox",
            "possible_polygon",
            "invalid_5_values",
            "other_or_invalid",
            "non_numeric",
        ):
            print(f"  {category}: {counts[category]}")
            if category in examples:
                print(f"    example: {examples[category]}")
