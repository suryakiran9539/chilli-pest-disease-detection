
from pathlib import Path

root = Path(r"D:\AI ml project data set\extracted_datasets")
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

checked = 0
empty = []
invalid = []
missing_images = []

for dataset in sorted(root.iterdir()):
    if not dataset.is_dir():
        continue

    print(f"\n===== {dataset.name} =====")

    for split in ["train", "valid", "test"]:
        image_dir = dataset / split / "images"
        label_dir = dataset / split / "labels"

        if not label_dir.exists():
            continue

        image_stems = (
            {p.stem for p in image_dir.iterdir()
             if p.is_file() and p.suffix.lower() in image_extensions}
            if image_dir.exists() else set()
        )

        for label_file in label_dir.glob("*.txt"):
            checked += 1

            if label_file.stem not in image_stems:
                missing_images.append(str(label_file))
                continue

            lines = label_file.read_text(
                encoding="utf-8-sig"
            ).splitlines()

            if not lines:
                empty.append(str(label_file))
                continue

            for line_number, line in enumerate(lines, start=1):
                parts = line.split()

                try:
                    values = [float(value) for value in parts]

                    valid_row = (
                        len(values) == 5
                        and values[0] >= 0
                        and values[0].is_integer()
                        and all(0 <= value <= 1 for value in values[1:])
                    )

                    if not valid_row:
                        raise ValueError("Invalid YOLO annotation")

                except ValueError:
                    invalid.append(
                        f"{label_file} (line {line_number}: {line})"
                    )

print("\n========== VALIDATION SUMMARY ==========")
print("Label files checked:", checked)
print("Empty label files:", len(empty))
print("Invalid annotations:", len(invalid))
print("Labels missing matching images:", len(missing_images))

for heading, items in [
    ("Empty labels", empty),
    ("Invalid annotations", invalid),
    ("Missing matching images", missing_images),
]:
    if items:
        print(f"\n{heading} (first 10):")
        for item in items[:10]:
            print(item)
