
from pathlib import Path
import shutil
import random
import yaml

ROOT = Path(r"D:\AI ml project data set\extracted_datasets")
DEST = Path(r"D:\AI ml project data set\combined_chilli_dataset")

DATASETS = [
    "chilli labelling 3.v2i.yolov11",
    "chilli project 4.v1i.yolov11",
    "chilli ventemole 1",
    "white fly 2",
]

random.seed(42)
SPLITS = {"train": 0.70, "valid": 0.20, "test": 0.10}


def load_yaml(path):
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or "names" not in data:
        raise ValueError(f"Invalid dataset YAML: {path}")
    names = data["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    return [str(name) for name in names]


def find_split_folder(base, split):
    candidates = [
        base / split,
        base / {"valid": "val", "train": "train", "test": "test"}[split],
    ]
    for candidate in candidates:
        if (candidate / "images").is_dir():
            return candidate
    return None


def convert_label(src, dst, class_map, num_classes):
    output = []
    if not src.exists():
        return output

    for line_no, line in enumerate(
        src.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        parts = line.split()
        if not parts:
            continue

        try:
            values = [float(v) for v in parts]
        except ValueError:
            raise ValueError(f"Non-numeric label: {src}, line {line_no}")

        old_id = int(values[0])
        if values[0] != old_id or old_id not in class_map:
            raise ValueError(f"Unknown class ID: {src}, line {line_no}")

        coords = values[1:]
        if len(coords) == 4:
            x, y, w, h = coords
            if w <= 0 or h <= 0:
                raise ValueError(f"Invalid box size: {src}, line {line_no}")
            if not all(0 <= v <= 1 for v in coords):
                raise ValueError(f"Coordinates out of range: {src}, line {line_no}")
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            # YOLO segmentation polygon: x1 y1 x2 y2 ...
            if not all(0 <= v <= 1 for v in coords):
                raise ValueError(f"Polygon coordinates out of range: {src}, line {line_no}")
            xs = coords[0::2]
            ys = coords[1::2]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            x, y = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                raise ValueError(f"Degenerate polygon: {src}, line {line_no}")
        else:
            raise ValueError(
                f"Unsupported annotation with {len(coords)} coordinates: "
                f"{src}, line {line_no}"
            )

        new_id = class_map[old_id]
        output.append(
            f"{new_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        )

    return output


def main():
    if not ROOT.is_dir():
        raise FileNotFoundError(f"Dataset folder missing: {ROOT}")

    # Read all source classes before copying anything.
    source_info = []
    unified_names = []

    for dataset in DATASETS:
        base = ROOT / dataset
        yaml_path = base / "data.yaml"
        if not yaml_path.is_file():
            raise FileNotFoundError(f"Missing YAML: {yaml_path}")

        names = load_yaml(yaml_path)
        source_info.append((dataset, base, names))

        for name in names:
            if name not in unified_names:
                unified_names.append(name)

    if DEST.exists():
        shutil.rmtree(DEST)

    total_images = 0
    total_labels = 0
    empty_labels = 0
    manifest = []

    for dataset, base, names in source_info:
        class_map = {i: unified_names.index(name) for i, name in enumerate(names)}
        for split in SPLITS:
            folder = find_split_folder(base, split)
            if folder is None:
                print(f"WARNING: {dataset} has no {split}/images folder")
                continue

            image_dir = folder / "images"
            label_dir = folder / "labels"
            images = sorted(
                p for p in image_dir.iterdir()
                if p.is_file() and p.suffix.lower() in
                {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            )

            for image_path in images:
                label_path = label_dir / (image_path.stem + ".txt")
                if not label_path.is_file():
                    raise FileNotFoundError(
                        f"Missing label for image: {image_path}"
                    )

                # Retain the source and split in each filename.
                safe_dataset = dataset.replace(" ", "_").replace(".", "_")
                unique_stem = f"{safe_dataset}_{split}_{image_path.stem}"
                out_image = DEST / split / "images" / f"{unique_stem}{image_path.suffix.lower()}"
                out_label = DEST / split / "labels" / f"{unique_stem}.txt"
                out_image.parent.mkdir(parents=True, exist_ok=True)
                out_label.parent.mkdir(parents=True, exist_ok=True)

                converted = convert_label(
                    label_path, out_label, class_map, len(unified_names)
                )
                shutil.copy2(image_path, out_image)
                out_label.write_text(
                    "\n".join(converted) + ("\n" if converted else ""),
                    encoding="utf-8",
                )

                total_images += 1
                total_labels += len(converted)
                if not converted:
                    empty_labels += 1
                manifest.append(
                    f"{split}\t{dataset}\t{image_path.name}\t{unique_stem}"
                )

    data = {
        "path": str(DEST),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(unified_names),
        "names": unified_names,
    }
    (DEST / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (DEST / "class_mapping.txt").write_text(
        "\n".join(f"{i}: {name}" for i, name in enumerate(unified_names)),
        encoding="utf-8",
    )
    (DEST / "manifest.tsv").write_text(
        "\n".join(manifest), encoding="utf-8"
    )

    print("\nDATASET PREPARATION COMPLETE")
    print("Classes:", len(unified_names))
    for i, name in enumerate(unified_names):
        print(f"  {i}: {name}")
    print("Images copied:", total_images)
    print("Bounding boxes written:", total_labels)
    print("Images with empty labels:", empty_labels)
    print("YAML:", DEST / "data.yaml")
    print("IMPORTANT: Review class_mapping.txt before training.")


if __name__ == "__main__":
    main()
