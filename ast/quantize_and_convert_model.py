# Based on multiple python files sent by Eddie including these: W_AST_TrainingWcomments.py and W_AST_Distill_To_EffNet.py
# Modified by Aaron Mathews
import litert_torch
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import tensorflow as tf

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")


# --- CNN Student Configuration
STUDENT_MODEL_NAME = 'efficientnet_b0'
DISTILLATION_TEMP = 2.0    # Temperature T for softening Teacher logits
DISTILLATION_ALPHA = 0.7 # Weight for KD loss (0.7 * KD Loss + 0.3 * BCE Loss)
TARGET_AUDIO_SECONDS = 10.24 # 10.24 seconds like AST
STUDENT_CHECKPOINT_PATH = "best_student_checkpoint.pt" # PATH TO SAVE THE TRAINED CNN STUDENT!!!!!!!!!<---------

class EfficientNetSpectrogramStudent(nn.Module):
    """
    EfficientNet model adapted for single-channel spectrogram input (B, 1, F, T)
    using torchvision models.
    """
    def __init__(self, model_name: str = 'efficientnet_b0', num_classes: int = 6):
        super().__init__()

        if model_name == 'efficientnet_b0':
            self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Model {model_name} not supported via torchvision in this script.")

        # 1. Adapt the first convolutional layer (conv_stem) for 1 channel
        original_conv = self.efficientnet.features[0][0]

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )

        # Initialize the new 1-channel weights by averaging the 3-channel weights
        with torch.no_grad():
            new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)

        self.efficientnet.features[0][0] = new_conv

        # 2. Adapt the classifier head
        num_in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_in_features, num_classes)

    def forward(self, input_values: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.efficientnet(input_values)

def validate(model, val_dataloader):
    model.eval()
    correct = 0
    total = 0

    predictions = []
    labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_data = {k: v.to(device) for k, v in batch.items()}

            # Use device.type as a POSITIONAL argument for autocast
            with autocast(device.type, dtype=torch.float16):
                student_logits = model(input_data['input_values'])

            curr_predictions = torch.argmax((torch.sigmoid(student_logits)), dim=1).long().flatten()
            curr_labels = input_data['labels'].long().flatten()

            predictions.append(curr_predictions.detach().cpu())
            labels.append(curr_labels.detach().cpu())

            for i in range(len(curr_predictions)):
                if curr_predictions[i] == curr_labels[i]:
                    correct += 1
            total += curr_labels.size(0)

    val_accuracy = correct / total if total > 0 else 0

    print(total)
    print(f"Accuracy: {val_accuracy}")

    c = confusion_matrix(torch.cat(labels).numpy(), torch.cat(predictions).numpy())
    data_labels = ["Negative", "Drone", "Piston", "Turbofan", "Turboprop", "Turboshaft"]
    sns.heatmap(c, annot=True, fmt='d', xticklabels=data_labels, yticklabels=data_labels)
    plt.title("Aircraft Classifier 2: Fine-tuning -> knowledge distillation -> AMP")
    plt.savefig("results.png", dpi=300, bbox_inches='tight')
    print(c)

    # Find precision and recall by class

    
    

    predictions = torch.cat(predictions)
    labels = torch.cat(labels)

    # We have 6 rows, each are formatted like this: [tp, fp, fn]
    precision_recall_array = np.zeros((6, 3))

    for i in range(len(predictions)):
        prediction = predictions[i].item()
        label = labels[i].item()
        if prediction == label:
            # If the prediction matches the label, then we have a true positive
            precision_recall_array[prediction][0] += 1
        else:
            # If the prediction and label don't match, then...
            # We have one more false positive for the predicted class
            # We have one more false negative for the labelled class
            precision_recall_array[prediction][1] += 1
            precision_recall_array[label][2] += 1

    print("Printing precision and recall by class \n")
    print(f"Negative class, precision: {precision_recall_array[0][0]/(precision_recall_array[0][0] + precision_recall_array[0][1])}, recall: {precision_recall_array[0][0]/(precision_recall_array[0][0] + precision_recall_array[0][2])} \n")
    print(f"Drone class, precision: {precision_recall_array[1][0]/(precision_recall_array[1][0] + precision_recall_array[1][1])}, recall: {precision_recall_array[1][0]/(precision_recall_array[1][0] + precision_recall_array[1][2])} \n")
    print(f"Piston class, precision: {precision_recall_array[2][0]/(precision_recall_array[2][0] + precision_recall_array[2][1])}, recall: {precision_recall_array[2][0]/(precision_recall_array[2][0] + precision_recall_array[2][2])} \n")
    print(f"Turbofan class, precision: {precision_recall_array[3][0]/(precision_recall_array[3][0] + precision_recall_array[3][1])}, recall: {precision_recall_array[3][0]/(precision_recall_array[3][0] + precision_recall_array[3][2])} \n")
    print(f"Turboprop class, precision: {precision_recall_array[4][0]/(precision_recall_array[4][0] + precision_recall_array[4][1])}, recall: {precision_recall_array[4][0]/(precision_recall_array[4][0] + precision_recall_array[4][2])} \n")
    print(f"Turboshaft class, precision: {precision_recall_array[5][0]/(precision_recall_array[5][0] + precision_recall_array[5][1])}, recall: {precision_recall_array[5][0]/(precision_recall_array[5][0] + precision_recall_array[5][2])} \n")


    # # Find micro averages
    # precision_micro_avg = 
    # recall_micro_avg = 0

    # # Find macro averages
    # precision_macro_avg = 0
    # recall_macro_avg = 0


if __name__ == '__main__':

    student_model = EfficientNetSpectrogramStudent(STUDENT_MODEL_NAME, num_classes=6)

    student_checkpoint_uncompressed = torch.load('best_student_checkpoint.pt')
    student_model.load_state_dict(student_checkpoint_uncompressed['model_state_dict'])

    student_model_eval_mode = student_model.eval()
    sample_input = (torch.randn(1, 1, 224, 224),)

    flags = {
        "optimizations": [tf.lite.Optimize.DEFAULT],
        "target_spec": {
            "supported_types": [tf.float16]
        }
    }

    tf_lite_model = litert_torch.convert(student_model_eval_mode, sample_input, _ai_edge_converter_flags=flags)
    tf_lite_model.export("quantized_model.tflite")
    
