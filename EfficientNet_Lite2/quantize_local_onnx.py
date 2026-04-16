import argparse
import random
from pathlib import Path

import numpy as np
import onnx
from PIL import Image
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "efficientnet_lite2_unquantized.onnx"
DEFAULT_CALIBRATION_PATH = PROJECT_DIR / "quantization_calibration_npzs" / "calibration_data.npz"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "efficientnet_lite2_locally_quantized.onnx"
DEFAULT_RAW_VAL_DIR = PROJECT_DIR / "ILSVRC2012_img_val"
IMG_SIZE = 260
DEFAULT_NUM_CALIBRATION_SAMPLES = 100
DEFAULT_RANDOM_SEED = 42


class NpzReader(CalibrationDataReader):
    def __init__(self, input_name, samples):
        self.input_name = input_name
        self.samples = iter(samples)

    def get_next(self):
        sample = next(self.samples, None)
        if sample is None:
            return None

        return {self.input_name: np.expand_dims(sample.astype(np.float32), axis=0)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Locally quantize the EfficientNet-Lite2 ONNX model using ONNX Runtime static PTQ."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the unquantized ONNX model.",
    )
    parser.add_argument(
        "--calibration",
        default=str(DEFAULT_CALIBRATION_PATH),
        help="Path to the calibration .npz file created from representative ImageNet samples.",
    )
    parser.add_argument(
        "--raw-val-dir",
        default=str(DEFAULT_RAW_VAL_DIR),
        help="Directory containing the flat ImageNet validation JPEG files used to build representative calibration samples when needed.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path where the locally quantized ONNX model should be written.",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=DEFAULT_NUM_CALIBRATION_SAMPLES,
        help="Number of representative validation images to use when building a calibration .npz locally.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used when selecting representative calibration images.",
    )
    return parser.parse_args()


def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")

    # Match the same local preprocessing path used for ImageNet evaluation and calibration
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype(np.float32)
    img /= 255.0
    img = (img - 0.5) / 0.5

    return img


def get_all_images(root_dir):
    image_paths = []

    for suffix in ("*.JPEG", "*.JPG", "*.jpeg", "*.jpg", "*.png"):
        image_paths.extend(root_dir.rglob(suffix))

    return sorted(image_paths)


def load_or_build_calibration_data(calibration_path, raw_val_dir, num_samples, random_seed):
    if calibration_path.exists():
        return np.load(calibration_path)["data"]

    if not raw_val_dir.exists():
        raise FileNotFoundError(
            f"Calibration file was not found at {calibration_path}, and raw ImageNet validation images were not found at {raw_val_dir}."
        )

    all_images = get_all_images(raw_val_dir)
    if len(all_images) < num_samples:
        raise ValueError(
            f"Requested {num_samples} calibration samples, but only found {len(all_images)} images in {raw_val_dir}."
        )

    # If the .npz is missing, build it here so the repo can still recreate the local PTQ flow
    random.seed(random_seed)
    selected_images = random.sample(all_images, num_samples)
    processed = [preprocess_image(img_path) for img_path in selected_images]
    calibration_data = np.stack(processed, axis=0).astype(np.float32)

    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(calibration_path, data=calibration_data)

    print(f"Built calibration dataset at: {calibration_path}")
    return calibration_data


def main():
    args = parse_args()

    model_path = Path(args.model)
    calibration_path = Path(args.calibration)
    output_path = Path(args.output)
    raw_val_dir = Path(args.raw_val_dir)

    model = onnx.load(model_path)
    input_name = model.graph.input[0].name
    samples = load_or_build_calibration_data(
        calibration_path=calibration_path,
        raw_val_dir=raw_val_dir,
        num_samples=args.num_calibration_samples,
        random_seed=args.random_seed,
    )

    # This keeps the local quantization path aligned with the ONNX Runtime PTQ settings used in the project
    quantize_static(
        model_input=str(model_path),
        model_output=str(output_path),
        calibration_data_reader=NpzReader(input_name, samples),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
    )

    print(f"Wrote quantized model to: {output_path}")


if __name__ == "__main__":
    main()
