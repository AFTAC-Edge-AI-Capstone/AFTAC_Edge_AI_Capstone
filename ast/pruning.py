from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch_pruning as tp
import torch
import torch.nn as nn


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



if __name__ == '__main__':
    model = EfficientNetSpectrogramStudent('efficientnet_b0', num_classes=6)
    example_inputs = torch.randn(1, 1, 224, 224)

    importance =  tp.importance.GroupMagnitudeImportance(p=2)

    pruner = tp.pruner.BasePruner(
        model,
        example_inputs,
        importance=importance,
        pruning_ratio=0.1
    )

    initial_macs, initial_params = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"Initial MACs: {initial_macs}, Initial Params: {initial_params}")
    pruner.step()
    final_macs, final_params = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"Final MACs: {final_macs}, Final Params: {final_params}")

    model.zero_grad()
    torch.save(model, 'pruned_model.pth')


