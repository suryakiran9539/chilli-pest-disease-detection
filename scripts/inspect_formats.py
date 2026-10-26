
from pathlib import Path

root = Path(r"D:\AI ml project data set\extracted_datasets")

for dataset in sorted(root.iterdir()):
    if not dataset.is_dir():
        continue

    print(f"\n{'=' * 55}")
    print("DATASET:", dataset.name)

    yaml_file = dataset / "data.yaml"
    if yaml_file.exists():
        for line in yaml_file.read_text(
            encoding="utf-8-sig"
        ).splitlines():
            if line.strip().startswith(("nc:", "names:")):
                print("YAML:", line)

    label_dir = dataset / "train" / "labels"
    files = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []

    print("Training label files:", len(files))

    if files:
        f = files[0]
        lines = f.read_text(encoding="utf-8-sig").splitlines()
        print("Sample label:", f.name)

        for line in lines[:2]:
            parts = line.split()
            print("Values in line:", len(parts))
            print("First values:", parts[:12])
