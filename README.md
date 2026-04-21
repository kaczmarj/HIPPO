# HIPPO

HIPPO is an explainability method and toolkit for weakly-supervised learning in computational pathology.

Please see our preprint on arXiv https://arxiv.org/abs/2409.03080.

> [!NOTE]
> This codebase is a work in progress. Please check back periodically for updates.

# Environment Setup

1. Git clone this repository
2. `cd HIPPO`
3. Create and activate the conda environment

```
conda create --name hippo-env python=3.10
conda activate hippo-env
```

4. Install the dependencies with

```
conda create -n hippo-env python=3.10
conda install openslide=3.4.1 openslide-python=1.3.1 -c conda-forge -y # openslide is best installed via conda
pip install torch==2.6 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

# How to use HIPPO

HIPPO is meant for weakly-supervised models in computational pathology. Before you use HIPPO, you need patch embeddings, and a trained attention-based multiple instance learning (ABMIL) model. Below, we briefly describe how to go from whole slide images (WSIs) to a trained ABMIL model.

We also made available models for metastasis detection, trained on CAMELYON16. Please see the following HuggingFace repositories for metastasis detection models trained using different encoders:
- UNI: https://huggingface.co/kaczmarj/metastasis-abmil-128um-uni
- REMEDIS: https://huggingface.co/kaczmarj/metastasis-abmil-128um-remedis
- Phikon: https://huggingface.co/kaczmarj/metastasis-abmil-128um-phikon
- CTransPath: https://huggingface.co/kaczmarj/metastasis-abmil-128um-ctranspath
- RetCCL: https://huggingface.co/kaczmarj/metastasis-abmil-128um-retccl

To simplify reproducibility, we also uploaded UNI embeddings for CAMELYON16 to https://huggingface.co/datasets/kaczmarj/camelyon16-uni. Embeddings using the other models may be uploaded in the future.

## Prepare your data for ABMIL

First separate your whole slide images into smaller, non-overlapping patches. The CLAM toolkit is one popular way to do this. After you have patch coordinates, you will have to encode those patches with a pre-trained model. There are countless options to choose from, but I would opt for a recent foundation model trained on a large and diverse set of histopathology images. Keep track of the patch coordinates and the patch features. This will be useful for downstream HIPPO experiments and visualizing attention maps.

# HIPPO Workflow

## 1. Extract Features from Foundation Model

Extract deep features from whole slide images using a foundation model encoder (e.g., Virchow2):

```bash
cd scripts
python extract_features.py \
  --encoder virchow2 \
  --wsi-dir /path/to/wsi/images/ \
  --patch-dir /path/to/patch/coordinates/ \
  --save-dir /path/to/output/features/ \
  --wsi-extension .tif \
  --batch-size 64 \
  --num-workers 8
```

**Parameters:**
- `--encoder`: Foundation model encoder to use (e.g., `virchow2`)
- `--wsi-dir`: Directory containing whole slide images
- `--patch-dir`: Directory containing patch coordinate HDF5 files
- `--save-dir`: Output directory for extracted features
- `--wsi-extension`: File extension of WSI files (e.g., `.svs`, `.ndpi`, `.tif`)
- `--batch-size`: Batch size for DataLoader (default: 64)
- `--num-workers`: Number of workers for data loading (default: 8)

## 2. Train ABMIL or VisionTransformer Models

Train an attention-based multiple instance learning (ABMIL) or Vision Transformer model on extracted features. The `../data/` directory contains the labels csv along with the json splits.

## Train ABMIL Model

```bash
cd scripts
python train_classification.py \
  --model-name VisionTransformer  \
  --features-dir /path/to/deep/features \
  --output-dir /path/to/output/directory \
  --csv ../data/camelyon16-labels.csv \
  --label-col binary_label_int \
  --num-classes 2 \
  --embedding-size 1024 \
  --split-json ../data/splits/camelyon16/camelyon16-split0.json \
  --fold 0 \
  --num-epochs 20 \
  --seed 0 \
  -L 512 \
  -D 384 \
  --lr 1e-4
```

**Common Parameters:**
- `--model-name`: Model architecture to train (choose one of `AttentionMILModel`, `AttentionMILMultiBranchModel`, `AdditiveAttentionMILModel`, `VisionTransformer`)
- `--features-dir`: Directory containing extracted features
- `--output-dir`: Directory to save trained model and results
- `--csv`: CSV file with slide labels and metadata
- `--label-col`: Column name in CSV containing labels (e.g. binary_label_int for camelyon16) 
- `--num-classes`: Number of classification classes
- `--embedding-size`: Dimension of feature embeddings (e.g., 1024 for Virchow2)
- `--split-json`: JSON file with train/validation/test splits
- `--fold`: Cross-validation fold number
- `--num-epochs`: Number of training epochs (default: 20)
- `--seed`: Random seed for reproducibility (default: 0)
- `--lr`: Learning rate (default: 1e-4)
- `-L`: Attention layer dimension (default: 256)
- `-D`: Attention module dimension (default: 256)
- `--dropout`: Dropout rate (default: 0.25)

## 3. Run HIPPO Search

Run the HIPPO algorithm to identify high-effect subsets of patches in a slide:

```bash
cd scripts
python run_hippo_search.py \
  --fold 0 \
  --features_root /path/to/deep/features \
  --slide_id a195bae3-357f-11eb-b1e7-001a7dda7111 \
  --model_root /path/to/trained/models/camelyon16/abmil-virchow2-128um_seed0/ \
  --output_dir /path/to/output/hippo/results/ \
  --optimizer minimize \
  --output_index_to_optimize 1
```

**Parameters:**
- `--fold`: Cross-validation fold number (default: 0)
- `--features_root`: Path to the root directory containing the features
- `--slide_id`: ID of the slide to analyze
- `--model_root`: Path to the root directory containing trained model artifacts
- `--output_dir`: Directory to save HIPPO search results
- `--optimizer`: Type of HIPPO optimization to perform (minimize, maximize, or smallest_difference)
- `output_index_to_optimize`: Index of the model output to optimize during the search (e.g., 1 for the positive class probability)

# Experimental Examples

For all examples, first enter the `scripts` directory by running `cd scripts`.

## Minimal reproducible example with synthetic data

The code below isn't intended to show any effect of an intervention. Rather, the purpose is to show how to use HIPPO to create an intervention in a specimen and evaluate the effects using a pretrained ABMIL model.

To work with real data and a pretrained model, see [the example below](#test-the-sufficiency-of-tumor-for-metastasis-detection).


```python
import models
import numpy as np
import torch

# Create the ABMIL model. Here, we use random initializations for the example.
# You should use a pretrained model in practice.
model = models.abmil.AttentionMILModel(in_features=1024, L=512, D=384, num_classes=2)
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

The code above should take less than 10 seconds to run.

## Test the sufficiency of tumor for metastasis detection

In the example below, we load a UNI-based ABMIL model for metastasis detection, trained on CAMELYON16.
Then, we take the embedding from one tumor patch from specimen `test_001` and add it to a negative specimen `test_003`.
The addition of this single tumor patch is enough to cause a positive metastasis result.

```python
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
```

The code above should take less than one minute to run.

## Test the effect of high attention regions

In this example, we evaluate the effect of high attention regions on metastasis detection. We find the following:

1. Using the original specimen, the model strongly predicts presence of metastasis (probability 0.997).
2. If we remove the top 1% of attended patches, the probability remains high for metastasis (0.988). This is presumably because some tumor patches remain in the specimen after removing top 1% of attention.
3. If we remove 5% of attention, then the probability of metastasis falls to 0.001.

In this way, we can quantify the effect of high attention regions.

```python
import math
import models
import huggingface_hub
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

The code above should take less than one minute to run.

## HIPPO greedy search algorithms

HIPPO implements greedy search algorithms to identify important patches. Below, we search for the patches that have the highest effect on metastasis detection. Briefly, we identify the patches that, when removed, result in the lowest probabilities for metastasis detections.

```python
import math
import models
import search
import huggingface_hub
import numpy as np
import torch

# Set our device.
device = torch.device("cpu")
# device = torch.device("cuda")  # Uncomment if you have a GPU.
# device = torch.device("mps")  # Uncomment if you have an ARM Apple computer.

# Load ABMIL model.
model = models.abmil.AttentionMILModel(in_features=1024, L=512, D=384, num_classes=2)
model.eval()
# You may need to run huggingface_hub.login() to get this file.
state_dict_path = huggingface_hub.hf_hub_download(
    "kaczmarj/metastasis-abmil-128um-uni", filename="seed2/model_best.pt"
)
state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model.to(device)

# Load features.
features_path = huggingface_hub.hf_hub_download(
    "kaczmarj/camelyon16-uni", filename="embeddings/test_064.pt", repo_type="dataset"
)
features = torch.load(features_path, weights_only=True).to(device)


# Define a function that takes in a bag of features and returns model probabilities.
# The output values are the values we want to optimize during our search.
# This is why we use a function -- models can have different outputs. By defining
# a function that returns the values we want to optimize on, we can streamline the code.
def model_probs_fn(features):
    with torch.inference_mode():
        logits, _ = model(features)
    # Shape of logits is 1xC, where C is number of classes.
    probs = logits.softmax(1).squeeze(0)  # C
    return probs


# Find the 1% highest effect patches. These are the patches that, when removed, drop the probability
# of metastasis the most. The `results` variable is a dictionary with.... results of the search!
# The model outputs in `results["model_outputs"]` correspond to the results after removing the patches
# in `results["ablated_patches"][:k]`.
num_rounds = math.ceil(len(features) * 0.01)
results = search.greedy_search(
    features=features,
    model_probs_fn=model_probs_fn,
    num_rounds=num_rounds,
    output_index_to_optimize=1,
    # We use minimize because we want to minimize the model outputs
    # when the patches are *removed*.
    optimizer=search.minimize,
)

# Now we can test the effect of removing the 1% highest effect patches.
patches_not_ablated = np.setdiff1d(np.arange(len(features)), results["ablated_patches"])
with torch.inference_mode():
    prob_baseline = model(features).logits.softmax(1)[0, 1].item()  # 1.000
    prob_without_high_effect = model(features[patches_not_ablated]).logits.softmax(1)[0, 1].item()  # 0.008

print(f"Probability of metastasis at baseline: {prob_baseline:0.3f}")
print(f"Probability of metastasis after removing 1% highest effect patches: {prob_without_high_effect:0.3f}")
```

The code above should take less than five minutes to run, with a GPU.

We can also plot the model outputs as we remove high effect patches, and we hope to see a monotonically decreasing line.

```python
import matplotlib.pyplot as plt
import numpy as np

model_results = results["model_outputs"][:, results["optimized_class_index"]]
plt.plot(model_results)
plt.xlabel("Number of patches removed")
plt.ylabel("Probability of metastasis")
```


# Cite

```bibtex
@article{kaczmarzyk2024explainable,
  title={Explainable AI for computational pathology identifies model limitations and tissue biomarkers},
  author={Kaczmarzyk, Jakub R and Kim, Chanwoo and Gadgil, Soham and Savant, Deepika and Zhao, Zhen and Saltz, Joel H and Lee, Su-In and Koo, Peter K},
  journal={arXiv preprint arXiv:2409.03080},
  year={2024}
}
```

# License

HIPPO code is licensed under the terms of the 3-Clause BSD License, and documentation is published under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International copyright license (CC BY-NC-SA 4.0).
