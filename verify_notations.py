
from pathlib import Path
from collections import Counter

p = Path(r"combined_chilli_dataset")
bad = []
empty = []
counts = {i: 0 for i in range(4)}
formats = Counter()
checked = 0

for split in ["train", "valid", "test"]:
    label_dir = p / split / "labels"

    for f in label_dir.glob("*.txt"):
        checked += 1
        lines = f.read_text(encoding="utf-8-sig").splitlines()

        if not any(line.strip() for line in lines):
            empty.append(str(f))
            continue

        for n, line in enumerate(lines, 1):
            v = line.split()

            try:
                a = [float(x) for x in v]

                if (
                    len(a) == 5
                    and a[0].is_integer()
                    and 0 <= int(a[0]) < 4
                    and all(0 <= x <= 1 for x in a[1:])
                ):
                    formats["bbox"] += 1
                    counts[int(a[0])] += 1

                elif (
                    len(a) > 5
                    and len(a) % 2 == 1
                    and a[0].is_integer()
                    and 0 <= int(a[0]) < 4
                ):
                    formats["possible_polygon"] += 1
                    bad.append(
                        f"{f} line {n}: possible polygon"
                    )
                else:
                    bad.append(
                        f"{f} line {n}: invalid format or class"
                    )

            except ValueError:
                bad.append(f"{f} line {n}: non-numeric")

print("Label files checked:", checked)
print("Annotation formats:", dict(formats))
print("Bounding boxes by class ID:", counts)
print("Empty label files:", len(empty))
print("Potential problems:", len(bad))
print("Examples:", bad[:10])