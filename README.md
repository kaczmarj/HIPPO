# HIPPO

HIPPO is an explainability toolkit for weakly-supervised learning in computational pathology.

Please see our preprint on arXiv https://arxiv.org/abs/2409.03080.

> [!NOTE]
> This codebase is a work in progress. Please check back periodically for updates.

Abstract
--------
<img src="docs/logo.png" width="200px" align="right" />

Deep learning models have shown promise in histopathology image analysis, but their opaque decision-making process poses challenges in high-risk medical scenarios. Here we introduce HIPPO, an explainable AI method that interrogates attention-based multiple instance learning (ABMIL) models in computational pathology by generating counterfactual examples through tissue patch modifications in whole slide images. Applying HIPPO to ABMIL models trained to detect breast cancer metastasis reveals that they may overlook small tumors and can be misled by non-tumor tissue, while attention maps—widely used for interpretation—often highlight regions that do not directly influence predictions. By interpreting ABMIL models trained on a prognostic prediction task, HIPPO identified tissue areas with stronger prognostic effects than high-attention regions, which sometimes showed counterintuitive influences on risk scores. These findings demonstrate HIPPO's capacity for comprehensive model evaluation, bias detection, and quantitative hypothesis testing. HIPPO greatly expands the capabilities of explainable AI tools to assess the trustworthy and reliable development, deployment, and regulation of weakly-supervised models in computational pathology.

If you find HIPPO useful, kindly [cite](#cite) it in your work.

# How to use HIPPO

HIPPO is meant for weakly-supervised, multiple instance learning models in computational pathology. Before you use HIPPO, you need patch embeddings, and and a trained attention-based multiple instance learning (ABMIL) model. Below, we briefly describe how to go from whole slide images (WSIs) to a trained ABMIL model.

We also made available models for metastasis detection, trained on CAMELYON16. Please see the following HuggingFace repositories for metastasis detection models trained using different encoders:
- UNI: https://huggingface.co/kaczmarj/metastasis-abmil-128um-uni
- REMEDIS: https://huggingface.co/kaczmarj/metastasis-abmil-128um-remedis
- Phikon: https://huggingface.co/kaczmarj/metastasis-abmil-128um-phikon
- CTransPath: https://huggingface.co/kaczmarj/metastasis-abmil-128um-ctranspath
- RetCCL: https://huggingface.co/kaczmarj/metastasis-abmil-128um-retccl

To simplify reproducibility, we also uploaded UNI embeddings for CAMELYON16 to https://huggingface.co/datasets/kaczmarj/camelyon16-uni. Embeddings using the other models may be uploaded in the future.

## Prepare your data for ABMIL

First separate your whole slide images into smaller, non-overlapping patches. The CLAM toolkit is one popular way to do this. After you have patch coordinates, you will have to encode those patches with a pre-trained model. There are countless options to choose from, but I would opt for a recent foundation model trained on a large and diverse set of histopathology images. Keep track of the patch coordinates and the patch features. This will be useful for downstream HIPPO experiments and visualizing attention maps.

## Train the ABMIL model

We provide a training script for classification models at https://huggingface.co/kaczmarj/metastasis-abmil-128um-uni/blob/main/train_classification.py. Alternatively, trained a model with [CLAM](https://github.com/mahmoodlab/CLAM) or another toolkit. HIPPO can work with any weakly-supervised model that accepts a bag of patches and returns a specimen-level output.

# Examples

## Minimal reproducubile example with synthetic data

The code below isn't intended to show any effect of an intervention. Rather, the purpose is to show how to use HIPPO to create an intervention in a specimen and evaluate the effects using a pretrained ABMIL model.

To work with real data and a pretrained model, see [the example below](#test-the-sufficiency-of-tumor-for-metastasis-detection).


```python
import hippo
import numpy as np
import torch

# Create the ABMIL model. Here, we use random initializations for the example.
# You should use a pretrained model in practice.
model = hippo.AttentionMILModel(in_features=1024, L=512, D=384, num_classes=2)
model.eval()

# We use random features. In practice, use actual features :)
features = torch.rand(1000, 1024)

# Define the intervention. Here, we want to remove five patches.
# We define the indices of the patches to keep.
patches_to_remove = np.array([500, 501, 502, 503, 504])
patches_to_keep = np.setdiff1d(np.arange(features.shape[0]), patches_to_remove)

# Get the model outputs for baseline and "treated" samples.
with torch.inference_mode():
    baseline = model(features).logits.softmax(1)
    treatment = model(features[patches_to_keep]).logits.softmax(1)
```

## Test the sufficiency of tumor for metastasis detection

In the example below, we load a UNI-based ABMIL model for metastasis detection, trained on CAMELYON16.
Then, we take the embedding from one tumor patch from specimen `test_001` and add it to a negative specimen `test_003`.
The addition of this single tumor patch is enough to cause a positive metastasis result.

```python
import hippo
import huggingface_hub
import numpy as np
import torch

# Create the ABMIL model. Here, we use random initializations for the example.
# You should use a pretrained model in practice.
model = hippo.AttentionMILModel(in_features=1024, L=512, D=384, num_classes=2)
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
```

## Test the effect of high attention regions

In this example, we evaluate the effect of high attention regions on metastasis detection. We find the following:

1. Using the original specimen, the model strongly predicts presence of metastasis (probability 0.997).
2. If we remove the top 1% of attended patches, the probability remains high for metastasis (0.988). This is presumably because some tumor patches remain in the specimen after removing top 1% of attention.
3. If we remove 5% of attention, then the probability of metastasis falls to 0.001.

In this way, we can quantify the effect of high attention regions.

```python
import math
import hippo
import huggingface_hub
import torch

# Create the ABMIL model. Here, we use random initializations for the example.
# You should use a pretrained model in practice.
model = hippo.AttentionMILModel(in_features=1024, L=512, D=384, num_classes=2)
model.eval()
# You may need to run huggingface_hub.login() to get this file.
state_dict_path = huggingface_hub.hf_hub_download(
    "kaczmarj/metastasis-abmil-128um-uni", filename="seed2/model_best.pt"
)
state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)

# Load features for positive specimen.
features_path = huggingface_hub.hf_hub_download(
    "kaczmarj/camelyon16-uni", filename="embeddings/test_001.pt", repo_type="dataset"
)
features = torch.load(features_path, weights_only=True)

# Get the model outputs for baseline and treated samples.
with torch.inference_mode():
    logits, attn = model(features)
attn = attn.squeeze(1).numpy()  # flatten tensor
tumor_prob = logits.softmax(1)[0, 1].item()
print(f"Tumor probability at baseline: {tumor_prob:0.3f}")

inds = attn.argsort()[::-1].copy()  # indices high to low, and copy to please torch
num_patches = math.ceil(len(inds) * 0.01)
with torch.inference_mode():
    logits_01pct, _ = model(features[inds[num_patches:]])
tumor_prob_01pct = logits_01pct.softmax(1)[0, 1].item()
print(f"Tumor probability after removing top 1% of attention: {tumor_prob_01pct:0.3f}")

num_patches = math.ceil(len(inds) * 0.05)
with torch.inference_mode():
    logits_05pct, _ = model(features[inds[num_patches:]])
tumor_prob_05pct = logits_05pct.softmax(1)[0, 1].item()
print(f"Tumor probability after removing top 5% of attention: {tumor_prob_05pct:0.3f}")
```

The following is printed:

```
Tumor probability at baseline: 0.997
Tumor probability after removing top 1% of attention: 0.988
Tumor probability after removing top 5% of attention: 0.001
```

# Cite

```bibtex
@misc{kaczmarzyk2024explainableaicomputationalpathology,
      title={Explainable AI for computational pathology identifies model limitations and tissue biomarkers},
      author={Jakub R. Kaczmarzyk and Joel H. Saltz and Peter K. Koo},
      year={2024},
      eprint={2409.03080},
      archivePrefix={arXiv},
      primaryClass={q-bio.TO},
      url={https://arxiv.org/abs/2409.03080},
}
```

# License

HIPPO code is licensed under the terms of the 3-Clause BSD License, and documentation is published under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International copyright license (CC BY-NC-SA 4.0).
