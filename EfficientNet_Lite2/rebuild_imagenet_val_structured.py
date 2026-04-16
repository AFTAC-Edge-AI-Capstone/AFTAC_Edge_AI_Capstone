import argparse
import csv
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_VAL_DIR = PROJECT_DIR / "ILSVRC2012_img_val"
DEFAULT_MANIFEST_PATH = PROJECT_DIR / "imagenet_val_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "imagenet_val_structured"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild the structured ImageNet validation directory from the saved image-to-synset manifest."
    )
    parser.add_argument(
        "--raw-val-dir",
        default=str(DEFAULT_RAW_VAL_DIR),
        help="Directory containing the flat ILSVRC2012 validation JPEG files.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="CSV manifest mapping image file names to synset folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the structured validation symlink tree should be rebuilt.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy images instead of creating symlinks.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    raw_val_dir = Path(args.raw_val_dir)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open(newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_name = row["image_name"]
            synset = row["synset"]
            source = raw_val_dir / image_name
            target_dir = output_dir / synset
            target = target_dir / image_name

            if not source.exists():
                raise FileNotFoundError(f"Missing validation image: {source}")

            target_dir.mkdir(parents=True, exist_ok=True)

            if target.exists() or target.is_symlink():
                target.unlink()

            if args.copy:
                target.write_bytes(source.read_bytes())
            else:
                target.symlink_to(source.resolve())

    print(f"Rebuilt structured validation directory at: {output_dir}")


if __name__ == "__main__":
    main()
