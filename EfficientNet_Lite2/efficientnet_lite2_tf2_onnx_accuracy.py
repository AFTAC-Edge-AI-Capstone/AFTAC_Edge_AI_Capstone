"""
This program loads an .onnx file storing a serialized computational graph + weights stored using Protocol Buffers
        think of a .onnx file as:
                                    Neural Network = Graph (structure) + Parameters (weights)
        an .onnx file contains both
    At a high level:
                    Model
                    ├── Graph
                    │    ├── Nodes (operations: Conv, Relu, Add, etc.)
                    │    ├── Edges (tensors flowing between nodes)
                    │    ├── Inputs
                    │    └── Outputs
                    └── Initializers (weights, biases)
"""

# Imports needed for the program
import onnx # Defines the "format"
import onnxruntime as ort
import numpy as np
import os
import json
import time
from PIL import Image


"""1. Deserialize the protobuf file into a Python object
    onnx.load returns:
                      deserialized_protobuf_model.graph.node
                      deserialized_protobuf_model.graph.initializer
                      deserialized_protobuf_model.graph.input
                      deserialized_protobuf_model.graph.output
    a.k.a a data structure representation of the graph
"""
# Previously ran efficientnet_lite2_tf2.onnx
deserialized_protobuf_model = onnx.load("efficientnet_lite2_locally_quantized.onnx")


"""2. Computation - ONNX Runtime
    the onnx library defines the "format", the onnx runtime library provides the "execution engine".
"""
session = ort.InferenceSession("efficientnet_lite2_locally_quantized.onnx")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Input name:", input_name)
print("Output name:", output_name)


"""3. Data preprocessing
    the model expects images of shape (1, 260, 260, 3) and values normalized to [-1, 1]
"""
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")

    # resize to match model input
    img = img.resize((260, 260))

    img = np.array(img).astype(np.float32)

    # normalize to [0, 1]
    img /= 255.0

    # normalize to [-1, 1]
    img = (img - 0.5) / 0.5

    return img


"""4. Load ImageNet validation images
    assumes images are named like ILSVRC2012_val_00000001.JPEG
    here we use the structured validation directory where the parent folder is the true synset
"""
structured_image_dir = "imagenet_val_structured"

if not os.path.isdir(structured_image_dir):
    raise FileNotFoundError(
        f"Expected to find '{structured_image_dir}' next to this script, but it was not found."
    )


"""5. Load ground truth labels
    the structured validation directory already gives the ground truth label directly
    parent folder name = true WordNet synset for every image inside that folder
    so each sample already comes in as:
                                  image path + correct synset label
"""
def load_structured_evaluation_samples(path):
    evaluation_samples = []

    for synset_name in sorted(os.listdir(path)):
        synset_dir = os.path.join(path, synset_name)

        if not os.path.isdir(synset_dir):
            continue

        for filename in sorted(os.listdir(synset_dir)):
            if filename.endswith(".JPEG"):
                evaluation_samples.append((os.path.join(synset_dir, filename), synset_name))

    return evaluation_samples


evaluation_samples = load_structured_evaluation_samples(structured_image_dir)
print("Evaluation source:", structured_image_dir)


"""6. Load class names (index → human-readable label)
    imagenet_class_index.json already gives both pieces we need:
                                                                 model output index → WordNet synset id
                                                                 model output index → human-readable class name
    that keeps the label logic in one place instead of spreading it across extra helper files
"""
def load_model_class_index(path):
    with open(path, "r") as f:
        class_index = json.load(f)

    model_index_to_wnid = []
    class_names = []

    for model_idx in range(len(class_index)):
        wnid, class_name = class_index[str(model_idx)]
        model_index_to_wnid.append(wnid)
        class_names.append(class_name.replace("_", " "))

    return model_index_to_wnid, class_names


"""6.1 Check that both label spaces cover the same 1000 classes
    the model class index and the structured validation directory should describe the exact same ImageNet-1k synset set
"""
def validate_label_space_alignment(model_index_to_wnid, evaluation_samples):
    model_wnids = set(model_index_to_wnid)
    label_wnids = {label_wnid for _, label_wnid in evaluation_samples}

    if model_wnids != label_wnids:
        raise ValueError("Model synset set does not match the structured validation synset set.")

    if len(model_index_to_wnid) != 1000:
        raise ValueError("Model class index should contain exactly 1000 entries.")


model_index_to_wnid, class_names = load_model_class_index("imagenet_class_index.json")
wnid_to_class_name = dict(zip(model_index_to_wnid, class_names))
validate_label_space_alignment(model_index_to_wnid, evaluation_samples)


"""6.2 Terminal output helpers
    every 10 images we print one compact progress block
    that block should answer the main ImageNet-1k questions immediately:
                                                                    top-1
                                                                    top-5
                                                                    running accuracy
                                                                    recent accuracy
                                                                    confidence + throughput
"""
def format_accuracy(correct, total):
    return f"{(correct / total) * 100:6.2f}% ({correct:>5}/{total:<5})"


def format_error(correct, total):
    return f"{(1.0 - (correct / total)) * 100:6.2f}%"


def format_top5_predictions(top5_indices, probabilities):
    formatted_predictions = []

    for rank, pred_idx in enumerate(top5_indices, start=1):
        pred_class = class_names[pred_idx]
        pred_wnid = model_index_to_wnid[pred_idx]
        pred_score = float(probabilities[pred_idx]) * 100.0
        formatted_predictions.append(
            f"{rank}. {pred_class} [{pred_wnid}] {pred_score:6.2f}%"
        )

    return formatted_predictions


def print_progress_block(
    start_idx,
    end_idx,
    total_images,
    interval_top1_correct,
    interval_top5_correct,
    interval_count,
    running_top1_correct,
    running_top5_correct,
    elapsed_interval,
    elapsed_total,
    last_sample_metrics,
):
    top5_lines = format_top5_predictions(
        last_sample_metrics["top5_indices"],
        last_sample_metrics["probabilities"],
    )

    print()
    print("=" * 108)
    print(f"ImageNet-1k Eval Progress | images {start_idx:>5}-{end_idx:<5} of {total_images}")
    print("-" * 108)
    print(f"{'Top-1 Accuracy (last 10)':<34} {format_accuracy(interval_top1_correct, interval_count)}")
    print(f"{'Top-5 Accuracy (last 10)':<34} {format_accuracy(interval_top5_correct, interval_count)}")
    print(f"{'Accuracy (running)':<34} {format_accuracy(running_top1_correct, end_idx)}")
    print(f"{'Top-5 Accuracy (running)':<34} {format_accuracy(running_top5_correct, end_idx)}")
    print(f"{'Top-1 Error (running)':<34} {format_error(running_top1_correct, end_idx)}")
    print(f"{'Top-5 Error (running)':<34} {format_error(running_top5_correct, end_idx)}")
    print(f"{'Images / sec (last 10)':<34} {interval_count / elapsed_interval:8.2f}")
    print(f"{'Images / sec (running)':<34} {end_idx / elapsed_total:8.2f}")
    print("-" * 108)
    print(f"{'Last image':<34} {os.path.basename(last_sample_metrics['img_path'])}")
    print(f"{'Label':<34} {last_sample_metrics['label_class']} [{last_sample_metrics['label_wnid']}]")
    print(f"{'Top-1 Prediction':<34} {last_sample_metrics['top1_class']} [{last_sample_metrics['top1_wnid']}] {last_sample_metrics['top1_score'] * 100.0:6.2f}%")
    print(f"{'Top-1 Correct?':<34} {'yes' if last_sample_metrics['top1_correct'] else 'no'}")
    print(f"{'Top-5 Correct?':<34} {'yes' if last_sample_metrics['top5_correct'] else 'no'}")
    print(f"{'Top-5 Predictions':<34} {top5_lines[0]}")

    for top5_line in top5_lines[1:]:
        print(f"{'':<34} {top5_line}")

    print("=" * 108)


"""6.3 Evaluation settings
    report_every controls how often the terminal gets a progress block
    with 10 here, every block summarizes the last 10 validation images
"""
report_every = 10
total_images = len(evaluation_samples)
print("Evaluation images:", total_images)
print("Report every:", report_every, "images")


"""7. Accuracy computation
    iterate through images, run inference, then compare inside one consistent label space
    the model prediction starts in TensorFlow output ordering
    the validation labels come straight from the structured folder names as WordNet synsets
    so we map the prediction into a WordNet synset first, then compare
    that keeps both sides in the same semantic space even when the numeric class orderings differ
"""
correct = 0
top5_correct = 0
total = 0
interval_top1_correct = 0
interval_top5_correct = 0
eval_start_time = time.time()
interval_start_time = eval_start_time


"""8. Evaluation loop
    IMPORTANT:
    the parent folder already gives the correct synset directly
"""
for i, (img_path, label_wnid) in enumerate(evaluation_samples):
    img = preprocess_image(img_path)

    # add batch dimension
    img = np.expand_dims(img, axis=0)

    output = session.run(None, {input_name: img})
    probabilities = output[0][0]
    top5_indices = np.argsort(probabilities)[-5:][::-1]
    pred = int(top5_indices[0])
    pred_wnid = model_index_to_wnid[pred]
    top5_wnids = [model_index_to_wnid[pred_idx] for pred_idx in top5_indices]
    top1_is_correct = pred_wnid == label_wnid
    top5_is_correct = label_wnid in top5_wnids

    # compare prediction and ground truth in the same synset space
    if top1_is_correct:
        correct += 1

    if top5_is_correct:
        top5_correct += 1

    interval_top1_correct += int(top1_is_correct)
    interval_top5_correct += int(top5_is_correct)

    # debug (first few samples only)
    if i < 5:
        print("Pred idx (model space):", pred,
              "| Pred class:", class_names[pred],
              "| Pred synset:", pred_wnid,
              "| Pred score:", f"{float(probabilities[pred]) * 100.0:.2f}%",
              "| Label synset:", label_wnid,
              "| Label class:", wnid_to_class_name[label_wnid])

    total += 1

    last_sample_metrics = {
        "img_path": img_path,
        "label_wnid": label_wnid,
        "label_class": wnid_to_class_name[label_wnid],
        "top1_class": class_names[pred],
        "top1_wnid": pred_wnid,
        "top1_score": float(probabilities[pred]),
        "top1_correct": top1_is_correct,
        "top5_correct": top5_is_correct,
        "top5_indices": top5_indices.tolist(),
        "probabilities": probabilities,
    }

    if total % report_every == 0 or total == total_images:
        interval_count = report_every if total % report_every == 0 else total % report_every
        elapsed_total = max(time.time() - eval_start_time, 1e-9)
        elapsed_interval = max(time.time() - interval_start_time, 1e-9)
        start_idx = total - interval_count + 1

        print_progress_block(
            start_idx=start_idx,
            end_idx=total,
            total_images=total_images,
            interval_top1_correct=interval_top1_correct,
            interval_top5_correct=interval_top5_correct,
            interval_count=interval_count,
            running_top1_correct=correct,
            running_top5_correct=top5_correct,
            elapsed_interval=elapsed_interval,
            elapsed_total=elapsed_total,
            last_sample_metrics=last_sample_metrics,
        )

        interval_top1_correct = 0
        interval_top5_correct = 0
        interval_start_time = time.time()


"""9. Final accuracy
"""
accuracy = correct / total
top5_accuracy = top5_correct / total
elapsed_total = max(time.time() - eval_start_time, 1e-9)

print()
print("=" * 108)
print("ImageNet-1k Eval Summary")
print("-" * 108)
print(f"{'Images evaluated':<34} {total}")
print(f"{'Top-1 Accuracy':<34} {accuracy * 100.0:6.2f}%")
print(f"{'Top-5 Accuracy':<34} {top5_accuracy * 100.0:6.2f}%")
print(f"{'Top-1 Error':<34} {(1.0 - accuracy) * 100.0:6.2f}%")
print(f"{'Top-5 Error':<34} {(1.0 - top5_accuracy) * 100.0:6.2f}%")
print(f"{'Elapsed time (sec)':<34} {elapsed_total:8.2f}")
print(f"{'Average images / sec':<34} {total / elapsed_total:8.2f}")
print("=" * 108)
