import models
import huggingface_hub
import numpy as np
import torch

# Create the ABMIL model. Here, we use random initializations for the example.
# You should use a pretrained model in practice.
model = models.abmil.AttentionMILModel(in_features=1024, L=512, D=384, num_classes=2)
model.eval()
# You may need to run huggingface_hub.login() to get this file.
state_dict_path = huggingface_hub.hf_hub_download(
    "kaczmarj/metastasis-abmil-128um-uni", filename="seed2/model_best.pt"
)
state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)

features_positive_path = huggingface_hub.hf_hub_download(
    "kaczmarj/camelyon16-uni", filename="embeddings/test_001.pt", repo_type="dataset"
)
features_positive = torch.load(features_positive_path, weights_only=True)
# This index contains the embedding for the tumor patch shown in Figure 2a of the HIPPO preprint.
tumor_patch = features_positive[7238].unsqueeze(0)  # 1x1024

features_negative_patch = huggingface_hub.hf_hub_download(
    "kaczmarj/camelyon16-uni", filename="embeddings/test_003.pt", repo_type="dataset"
)
features_negative = torch.load(features_negative_patch, weights_only=True)

# Get the model outputs for baseline and treated samples.
with torch.inference_mode():
    baseline = model(features_negative).logits.softmax(1)[0, 1].item()
    treatment = model(torch.cat([features_negative, tumor_patch])).logits.softmax(1)[0, 1].item()

print(f"Probability of tumor in baseline: {baseline:0.3f}")  # 0.002
print(f"Probability of tumor after adding one tumor patch: {treatment:0.3f}")  # 0.824