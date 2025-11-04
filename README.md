<img src="images/logo.png" align=mid />

# MedicalNet for huggingface

The MedicalNet project aggregated the dataset with diverse modalities, target organs, and pathologies to to build relatively large datasets. Based on this dataset, a series of 3D-ResNet pre-trained models and corresponding transfer-learning training code are provided. 

This repository is an unofficial implementation of Tencent's Med3D model ([Med3D: Transfer Learning for 3D Medical Image Analysis](https://arxiv.org/abs/1904.00625)), originally developed for 3d segmentation tasks.
It has been adapted for classification tasks using the 3D-ResNet backbone and made compatible with the Hugging Face library.

---

## License
MedicalNet is released under the MIT License (refer to the LICENSE file for details).

---

## Citing MedicalNet
If you use this code or pre-trained models, please cite the following:
```
    @article{chen2019med3d,
        title={Med3D: Transfer Learning for 3D Medical Image Analysis},
        author={Chen, Sihong and Ma, Kai and Zheng, Yefeng},
        journal={arXiv preprint arXiv:1904.00625},
        year={2019}
    }
```

---

## Model Sources

- Repository: https://github.com/Tencent/MedicalNet (original)
- Unofficial Torch Hub Wrapper: https://github.com/Warvito/MedicalNet-models

---

## Usage

### 1. Download Pre-trained Models

First, download the original MedicalNet model weights and store them in the `resnet_pth/`directory.

### 2. Upload Models to Hugging Face Hub

To upload the models, you must log in first:

```bash
huggingface-cli login
```

#### Upload a Single Model

```bash
uv run python upload_resnet_to_hub.py \
    --model_variant resnet50 \
    --model_name "your-username/medicalnet-resnet3d50" \
```

#### Upload All Models

```bash
uv run python upload_resnet_to_hub.py \
    --upload_all \
    --username "your-username" \
```

### 3. Use the Model from Hugging Face Hub

To use the uploaded model:

```python
from transformers import AutoConfig, AutoModelForImageClassification
import torch

config = AutoConfig.from_pretrained(
    'nwirandx/medicalnet-resnet3d50',
    trust_remote_code=True
)

# use a model from scratch
# model = AutoModelForImageClassification.from_config(
#     config,
#     trust_remote_code=True
# )

# use pretrained model
model = AutoModelForImageClassification.from_pretrained(
    'nwirandx/medicalnet-resnet3d50',
    trust_remote_code=True
)

x = torch.randn(1, 1, 64, 64, 64)  # Example 3D volume
outputs = model(x)
print(outputs.logits.shape)  # (1, num_labels)
```

---

## MedicalNet Model Family

**Original MedicalNet Series (Tencent on Hugging Face)**

- [TencentMedicalNet/MedicalNet-Resnet10](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet10)
- [TencentMedicalNet/MedicalNet-Resnet18](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18)
- [TencentMedicalNet/MedicalNet-Resnet34](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet34)
- [TencentMedicalNet/MedicalNet-Resnet50](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet50)
- [TencentMedicalNet/MedicalNet-Resnet101](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet101)
- [TencentMedicalNet/MedicalNet-Resnet152](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet152)
- [TencentMedicalNet/MedicalNet-Resnet200](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet200)

**Unonfficial Versions of the MedcialNet Classifcation Model Series**

- [nwirandx/medicalnet-resnet3d10](https://huggingface.co/nwirandx/medicalnet-resnet3d10)
- [nwirandx/medicalnet-resnet3d10_23datasets](https://huggingface.co/nwirandx/medicalnet-resnet3d10_23datasets)
- [nwirandx/medicalnet-resnet3d50](https://huggingface.co/nwirandx/medicalnet-resnet3d50)
- [nwirandx/medicalnet-resnet3d50_23datasets](https://huggingface.co/nwirandx/medicalnet-resnet3d50_23datasets)
- [nwirandx/medicalnet-resnet3d101](https://huggingface.co/nwirandx/medicalnet-resnet3d101)
- [nwirandx/medicalnet-resnet3d152](https://huggingface.co/nwirandx/medicalnet-resnet3d152)
- [nwirandx/medicalnet-resnet3d200](https://huggingface.co/nwirandx/medicalnet-resnet3d200)
