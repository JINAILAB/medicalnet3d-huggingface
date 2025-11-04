"""
Upload MedicalNet ResNet3D models to Hugging Face Hub

Usage:
    # Upload single model
    python upload_resnet_to_hub.py --model_variant resnet10 --model_name "your-username/medicalnet-resnet3d-10"
    
    # Upload all models automatically
    python upload_resnet_to_hub.py --upload_all --username "your-username"

Example:
    python upload_resnet_to_hub.py --model_variant resnet50 --model_name "myuser/medicalnet-resnet3d-50"
"""

import argparse
import os
import shutil
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from safetensors.torch import save_file

from resnet_model.configuration_resnet import (
    ResNet3DConfig,
    ResNet3D10Config,
    ResNet3D50Config,
    ResNet3D101Config,
    ResNet3D152Config,
    ResNet3D200Config,
)
from resnet_model.modeling_resnet import (
    ResNet3DModel,
    ResNet3DForImageClassification,
    ResNet3D10ForImageClassification,
    ResNet3D50ForImageClassification,
    ResNet3D101ForImageClassification,
    ResNet3D152ForImageClassification,
    ResNet3D200ForImageClassification,
)


# MedicalNet 모델 정보
MEDICALNET_MODELS = {
    "10": {
        "filename": "resnet_10.pth",
        "local_path": "resnet_pth/resnet_10.pth",
        "config_class": ResNet3D10Config,
        "model_class": ResNet3D10ForImageClassification,
        "depths": [1, 1, 1, 1],
        "layer_type": "basic",
        "description": "MedicalNet ResNet3D-10 pretrained on medical dataset",
    },
    "10-23datasets": {
        "filename": "resnet_10_23dataset.pth",
        "local_path": "resnet_pth/resnet_10_23dataset.pth",
        "config_class": ResNet3D10Config,
        "model_class": ResNet3D10ForImageClassification,
        "depths": [1, 1, 1, 1],
        "layer_type": "basic",
        "description": "MedicalNet ResNet3D-10 pretrained on 23 medical datasets",
    },
    "50": {
        "filename": "resnet_50.pth",
        "local_path": "resnet_pth/resnet_50.pth",
        "config_class": ResNet3D50Config,
        "model_class": ResNet3D50ForImageClassification,
        "depths": [3, 4, 6, 3],
        "layer_type": "bottleneck",
        "description": "MedicalNet ResNet3D-50 pretrained on medical dataset",
    },
    "50-23datasets": {
        "filename": "resnet_50_23dataset.pth",
        "local_path": "resnet_pth/resnet_50_23dataset.pth",
        "config_class": ResNet3D50Config,
        "model_class": ResNet3D50ForImageClassification,
        "depths": [3, 4, 6, 3],
        "layer_type": "bottleneck",
        "description": "MedicalNet ResNet3D-50 pretrained on 23 medical datasets",
    },
    "101": {
        "filename": "resnet_101.pth",
        "local_path": "resnet_pth/resnet_101.pth",
        "config_class": ResNet3D101Config,
        "model_class": ResNet3D101ForImageClassification,
        "depths": [3, 4, 23, 3],
        "layer_type": "bottleneck",
        "description": "MedicalNet ResNet3D-101 pretrained model",
    },
    "152": {
        "filename": "resnet_152.pth",
        "local_path": "resnet_pth/resnet_152.pth",
        "config_class": ResNet3D152Config,
        "model_class": ResNet3D152ForImageClassification,
        "depths": [3, 8, 36, 3],
        "layer_type": "bottleneck",
        "description": "MedicalNet ResNet3D-152 pretrained model",
    },
    "200": {
        "filename": "resnet_200.pth",
        "local_path": "resnet_pth/resnet_200.pth",
        "config_class": ResNet3D200Config,
        "model_class": ResNet3D200ForImageClassification,
        "depths": [3, 24, 36, 3],
        "layer_type": "bottleneck",
        "description": "MedicalNet ResNet3D-200 pretrained model",
    },
}


def get_model_path(local_path: str) -> str:
    """Check local model file path"""
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Model file not found: {local_path}")
    
    file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"   Model file found: {os.path.basename(local_path)} ({file_size_mb:.1f} MB)")
    return local_path


def convert_old_keys_to_new(old_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert old MedicalNet model keys to new HuggingFace style keys
    
    Old structure:
    - conv1, bn1 -> resnet3d.embedder.embedder.convolution, normalization
    - maxpool -> resnet3d.embedder.pooler
    - layer1, layer2, layer3, layer4 -> resnet3d.encoder.stages[0-3]
    - avgpool -> resnet3d.pooler
    - fc -> classifier.1
    """
    new_state_dict = {}
    
    for old_key, value in old_state_dict.items():
        new_key = old_key
        
        # conv1 -> embedder.embedder.convolution
        if old_key == "conv1.weight":
            new_key = "resnet3d.embedder.embedder.convolution.weight"
        elif old_key == "conv1.bias":
            new_key = "resnet3d.embedder.embedder.convolution.bias"
        
        # bn1 -> embedder.embedder.normalization
        elif old_key.startswith("bn1."):
            param_name = old_key.replace("bn1.", "")
            new_key = f"resnet3d.embedder.embedder.normalization.{param_name}"
        
        # layer1-4 -> encoder.stages[0-3]
        elif old_key.startswith("layer"):
            # layer1 -> stage 0, layer2 -> stage 1, etc.
            parts = old_key.split(".")
            layer_num = int(parts[0].replace("layer", ""))
            stage_idx = layer_num - 1
            
            # layer1.0.conv1 -> encoder.stages[0].layers.0.layer.0.convolution
            block_idx = parts[1]
            rest = ".".join(parts[2:])
            
            # BasicBlock: conv1, bn1, conv2, bn2, downsample
            # Bottleneck: conv1, bn1, conv2, bn2, conv3, bn3, downsample
            
            if rest.startswith("downsample."):
                # downsample.0 -> shortcut.convolution
                # downsample.1 -> shortcut.normalization
                if "0.weight" in rest or "0.bias" in rest:
                    param = rest.split(".")[-1]
                    new_key = f"resnet3d.encoder.stages.{stage_idx}.layers.{block_idx}.shortcut.convolution.{param}"
                else:
                    param_name = rest.replace("downsample.1.", "")
                    new_key = f"resnet3d.encoder.stages.{stage_idx}.layers.{block_idx}.shortcut.normalization.{param_name}"
            
            elif rest.startswith("conv1"):
                # conv1 -> layer.0.convolution (for BasicBlock) or layer.0.convolution (for Bottleneck)
                param = rest.replace("conv1.", "")
                new_key = f"resnet3d.encoder.stages.{stage_idx}.layers.{block_idx}.layer.0.convolution.{param}"
            
            elif rest.startswith("bn1"):
                # bn1 -> layer.0.normalization
                param = rest.replace("bn1.", "")
                new_key = f"resnet3d.encoder.stages.{stage_idx}.layers.{block_idx}.layer.0.normalization.{param}"
            
            elif rest.startswith("conv2"):
                # conv2 -> layer.1.convolution
                param = rest.replace("conv2.", "")
                new_key = f"resnet3d.encoder.stages.{stage_idx}.layers.{block_idx}.layer.1.convolution.{param}"
            
            elif rest.startswith("bn2"):
                # bn2 -> layer.1.normalization
                param = rest.replace("bn2.", "")
                new_key = f"resnet3d.encoder.stages.{stage_idx}.layers.{block_idx}.layer.1.normalization.{param}"
            
            elif rest.startswith("conv3"):
                # conv3 -> layer.2.convolution (only for Bottleneck)
                param = rest.replace("conv3.", "")
                new_key = f"resnet3d.encoder.stages.{stage_idx}.layers.{block_idx}.layer.2.convolution.{param}"
            
            elif rest.startswith("bn3"):
                # bn3 -> layer.2.normalization (only for Bottleneck)
                param = rest.replace("bn3.", "")
                new_key = f"resnet3d.encoder.stages.{stage_idx}.layers.{block_idx}.layer.2.normalization.{param}"
        
        # fc -> classifier.1
        elif old_key.startswith("fc."):
            param = old_key.replace("fc.", "")
            new_key = f"classifier.1.{param}"
        
        new_state_dict[new_key] = value
    
    return new_state_dict


_MODELS_REGISTERED = False

def register_resnet3d_models():
    """Register ResNet3D models to AutoClass"""
    global _MODELS_REGISTERED
    
    if _MODELS_REGISTERED:
        return
    
    # Register AutoConfig
    AutoConfig.register("resnet3d", ResNet3DConfig)
    
    # Register AutoModel
    AutoModel.register(ResNet3DConfig, ResNet3DModel)
    
    # Register AutoModelForImageClassification
    from transformers import AutoModelForImageClassification
    AutoModelForImageClassification.register(ResNet3DConfig, ResNet3DForImageClassification)
    
    _MODELS_REGISTERED = True
    print(" ResNet3D models registered to AutoClass")


def load_pretrained_weights(model, pth_file: str):
    """Load pretrained weights into model and convert to safetensors"""
    device = torch.device("cpu")  # Load on CPU to save memory
    
    print(f"  📥 Loading PTH file...")
    pretrained_state_dict = torch.load(pth_file, map_location=device)
    
    # Clean state_dict keys
    if "state_dict" in pretrained_state_dict:
        pretrained_state_dict = pretrained_state_dict["state_dict"]
    
    # Remove DataParallel wrapper
    pretrained_state_dict = {k.replace("module.", ""): v for k, v in pretrained_state_dict.items()}
    
    print(f"  🔄 Converting keys (Original MedicalNet -> HuggingFace style)...")
    # Convert keys
    converted_state_dict = convert_old_keys_to_new(pretrained_state_dict)
    
    # Get current model's state_dict
    model_state_dict = model.state_dict()
    
    # Load only matching keys
    matched_keys = []
    mismatched_keys = []
    missing_keys = []
    
    for key in converted_state_dict.keys():
        if key in model_state_dict:
            if converted_state_dict[key].shape == model_state_dict[key].shape:
                matched_keys.append(key)
            else:
                mismatched_keys.append(key)
                print(f"     - Shape mismatch: {key}")
                print(f"     - Pretrained: {converted_state_dict[key].shape}")
                print(f"     - Current model: {model_state_dict[key].shape}")
    
    # New keys only in model (classification head, etc.)
    for key in model_state_dict.keys():
        if key not in converted_state_dict:
            missing_keys.append(key)
    
    # Load only matching weights
    filtered_state_dict = {k: v for k, v in converted_state_dict.items() if k in matched_keys}
    model.load_state_dict(filtered_state_dict, strict=False)
    
    print(f"   Weights loaded successfully:")
    print(f"     - Loaded layers: {len(matched_keys)}")
    print(f"     - Newly initialized layers: {len(missing_keys)}")
    if mismatched_keys:
        print(f"     - Excluded due to shape mismatch: {len(mismatched_keys)}")
    
    if len(matched_keys) < 10:
        print(f"\n   Warning: Loaded layers are very few. Check key mapping.")
        print(f"  mple original keys: {list(pretrained_state_dict.keys())[:3]}")
        print(f"   Sample converted keys: {list(converted_state_dict.keys())[:3]}")
        print(f"   Sample model keys: {list(model_state_dict.keys())[:3]}")
    
    return model


def create_model_readme(model_variant: str, model_name: str) -> str:
    """
    Create README.md content for the model card
    
    Args:
        model_variant: Model variant (e.g. '10', '50-23datasets')
        model_name: Model name to upload to Hub (e.g. "username/medicalnet-resnet3d-10")
    
    Returns:
        README content as string
    """
    model_info = MEDICALNET_MODELS[model_variant]
    
    # base_model 매핑 (Tencent 공식 모델)
    base_model_map = {
        "10": "TencentMedicalNet/MedicalNet-Resnet10",
        "10-23datasets": "TencentMedicalNet/MedicalNet-Resnet10",
        "50": "TencentMedicalNet/MedicalNet-Resnet50",
        "50-23datasets": "TencentMedicalNet/MedicalNet-Resnet50",
        "101": "TencentMedicalNet/MedicalNet-Resnet101",
        "152": "TencentMedicalNet/MedicalNet-Resnet152",
        "200": "TencentMedicalNet/MedicalNet-Resnet200",
    }
    
    base_model = base_model_map.get(model_variant, "TencentMedicalNet/MedicalNet-Resnet10")
    
    readme_content = f"""---
library_name: transformers
tags:
- MedicalNet
- medical images
- medical
- 3D
- Med3D
license: mit
datasets:
- TencentMedicalNet/MRBrains18
language:
- en
base_model:
- {base_model}
thumbnail: "https://github.com/Tencent/MedicalNet/blob/master/images/logo.png?raw=true"

---
# MedicalNet for classification

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
    @article{{chen2019med3d,
        title={{Med3D: Transfer Learning for 3D Medical Image Analysis}},
        author={{Chen, Sihong and Ma, Kai and Zheng, Yefeng}},
        journal={{arXiv preprint arXiv:1904.00625}},
        year={{2019}}
    }}
```

---

## Model Sources

- Repository: https://github.com/Tencent/MedicalNet (original)
- Unofficial Torch Hub Wrapper: https://github.com/Warvito/MedicalNet-models
- Unofficial Huggingface Wrapper: https://github.com/JINAILAB/medicalnet3d-huggingface

---

## How to Get Started with the Model

```python
from transformers import AutoConfig, AutoModelForImageClassification
import torch

config = AutoConfig.from_pretrained(
    '{model_name}',
    trust_remote_code=True
)

# use a model from scratch
# model = AutoModelForImageClassification.from_config(
#     config,
#     trust_remote_code=True
# )

# use pretrained model
model = AutoModelForImageClassification.from_pretrained(
    '{model_name}',
    trust_remote_code=True
)

x = torch.randn(1, 1, 64, 64, 64)  # Example 3D volume
outputs = model(x)
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

**Unofficial Versions of the MedicalNet Classification Model Series**

- [nwirandx/medicalnet-resnet3d10](https://huggingface.co/nwirandx/medicalnet-resnet3d10)
- [nwirandx/medicalnet-resnet3d10-23datasets](https://huggingface.co/nwirandx/medicalnet-resnet3d10-23datasets)
- [nwirandx/medicalnet-resnet3d50](https://huggingface.co/nwirandx/medicalnet-resnet3d50)
- [nwirandx/medicalnet-resnet3d50-23datasets](https://huggingface.co/nwirandx/medicalnet-resnet3d50-23datasets)
- [nwirandx/medicalnet-resnet3d101](https://huggingface.co/nwirandx/medicalnet-resnet3d101)
- [nwirandx/medicalnet-resnet3d152](https://huggingface.co/nwirandx/medicalnet-resnet3d152)
- [nwirandx/medicalnet-resnet3d200](https://huggingface.co/nwirandx/medicalnet-resnet3d200)
"""
    
    return readme_content


def upload_model_to_hub(
    model_variant: str,
    model_name: str,
    spatial_dims: int = 3,
    num_channels: int = 1,
    num_labels: int = 2,
):
    """
    Upload MedicalNet ResNet3D model to Hugging Face Hub
    
    Args:
        model_variant: Model variant (e.g. 'resnet10', 'resnet50_23datasets')
        model_name: Model name to upload to Hub (e.g. "username/medicalnet-resnet3d-10")
        spatial_dims: Spatial dimensions (3D medical images so 3)
        num_channels: Input channels
        num_labels: Output classes
    """
    print("=" * 80)
    print(f"Uploading MedicalNet {model_variant.upper()} model to Hugging Face Hub.")
    print("=" * 80)
    
    if model_variant not in MEDICALNET_MODELS:
        raise ValueError(f"Unsupported model variant: {model_variant}")
    
    model_info = MEDICALNET_MODELS[model_variant]
    
    # Check local model file
    print(f"\n Checking local model file...")
    pth_file = get_model_path(model_info["local_path"])
    
    # Create Configuration
    print(f"\n Creating Configuration...")
    config_class = model_info["config_class"]
    config = config_class(
        spatial_dims=spatial_dims,
        num_channels=num_channels,
        num_labels=num_labels,
    )
    
    print(f"  - Model: ResNet3D-{model_variant}")
    print(f"  - Spatial Dimensions: {config.spatial_dims}D")
    print(f"  - Input Channels: {config.num_channels}")
    print(f"  - Output Classes: {config.num_labels}")
    print(f"  - Depths: {config.depths}")
    print(f"  - Layer Type: {config.layer_type}")
    
    # Create model
    print(f"\n  Creating model...")
    model_class = model_info["model_class"]
    model = model_class(config)
    
    # Load pretrained weights
    print(f"\n  Loading pretrained weights...")
    model = load_pretrained_weights(model, pth_file)
    
    # Save model to temporary directory and copy code files
    print(f"\n Saving model to local directory...")
    temp_dir = f"./temp_{model_variant}"
    
    # Delete temporary directory if it exists
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # Save model and configuration
    model.save_pretrained(temp_dir, safe_serialization=True)
    config.save_pretrained(temp_dir)
    print(f"  Model and configuration saved successfully: {temp_dir}")
    
    # Copy model code files (required for trust_remote_code)
    print(f"\n Copying model code files...")
    source_config_file = "resnet_model/configuration_resnet.py"
    source_modeling_file = "resnet_model/modeling_resnet.py"
    
    shutil.copy2(source_config_file, os.path.join(temp_dir, "configuration_resnet.py"))
    shutil.copy2(source_modeling_file, os.path.join(temp_dir, "modeling_resnet.py"))
    print(f"  configuration_resnet.py copied successfully")
    print(f"  modeling_resnet.py copied successfully")
    
    # Create and save README.md
    print(f"\n Creating README.md...")
    readme_content = create_model_readme(model_variant, model_name)
    readme_path = os.path.join(temp_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"  README.md created successfully")
    
    # Upload to Hugging Face Hub
    print(f"\n Uploading to Hugging Face Hub...")
    print(f"  - Model Name: {model_name}")
    print(f"  - Description: {model_info['description']}")
    print(f"  - Format: safetensors")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Create repository (if it already exists, ignore)
        print(f"\n Checking/creating repository...")
        try:
            api.create_repo(
                repo_id=model_name,
                repo_type="model",
                exist_ok=True,  # If it already exists, ignore
                private=False
            )
            print(f"   Repository prepared successfully")
        except Exception as e:
            print(f"   Repository creation warning: {e}")
            print(f"   Trying to upload to existing repository...")
        
        print(f"\n Uploading entire folder...")
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=model_name,
            repo_type="model",
            commit_message=f"Upload {model_variant} model with trust_remote_code support"
        )
        print(f"   Upload completed")
        
        # Delete temporary directory
        print(f"\n Deleting temporary directory...")
        shutil.rmtree(temp_dir)
        print(f"   Temporary directory deleted")
        
        print(f"\n" + "=" * 80)
        print(f"Upload completed!")
        
        return True
        
    except Exception as e:
        print(f"\n Upload failed: {e}")
        print(f"\n Check the following:")
        print(f"  1. Check if you are logged in to Hugging Face")
        print(f"     Run in terminal: huggingface-cli login")
        print(f"  2. Check if the model name is in the correct format (username/model-name)")
        print(f"  3. Check network connection status")
        
        # Delete temporary directory if error occurs
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        raise


def upload_all_models(username: str, num_labels: int = 400):
    """Uploading all MedicalNet models to Hub"""
    print("\n" + "=" * 80)
    print("Uploading all MedicalNet ResNet3D models")
    print("=" * 80)
    
    results = {}
    
    for variant_name in MEDICALNET_MODELS.keys():
        model_name = f"{username}/medicalnet-resnet3d{variant_name.replace('_', '-')}"
        print(f"\n\n{'='*80}")
        print(f"[{list(MEDICALNET_MODELS.keys()).index(variant_name) + 1}/{len(MEDICALNET_MODELS)}] {variant_name} 업로드 시작")
        print(f"{'='*80}")
        
        try:
            success = upload_model_to_hub(
                model_variant=variant_name,
                model_name=model_name,
                num_labels=num_labels,
            )
            results[variant_name] = " 성공"
        except Exception as e:
            print(f"❌ {variant_name} 업로드 실패: {e}")
            results[variant_name] = f"❌ 실패: {str(e)[:50]}"
            continue
    
    # Print final results
    print("\n\n" + "=" * 80)
    print("Upload results summary")
    print("=" * 80)
    for variant, status in results.items():
        print(f"  {variant:25s} : {status}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="MedicalNet ResNet3D 모델을 Hugging Face Hub에 업로드"
    )
    
    # 단일 모델 업로드 옵션
    parser.add_argument(
        "--model_variant",
        type=str,
        choices=list(MEDICALNET_MODELS.keys()),
        help="Model variant to upload (e.g. 'resnet10', 'resnet50_23datasets')",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name to upload to Hub (e.g. 'username/medicalnet-resnet3d-10')",
    )
    
    # 전체 모델 업로드 옵션
    parser.add_argument(
        "--upload_all",
        action="store_true",
        help="Upload all MedicalNet models automatically",
    )
    parser.add_argument(
        "--username",
        type=str,
        help="Hugging Face username (--upload_all required)",
    )
    
    # 공통 옵션
    parser.add_argument(
        "--spatial_dims",
        type=int,
        default=3,
        help="Spatial dimensions (default: 3)",
    )
    parser.add_argument(
        "--num_channels",
        type=int,
        default=1,
        help="Input channels (default: 1)",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=2,
        help="Output classes (default: 400, MedicalNet pretrained)",
    )
    
    args = parser.parse_args()
    
    # 사용 가능한 모델 목록 출력
    print("\nAvailable MedicalNet models:")
    for variant, info in MEDICALNET_MODELS.items():
        print(f"  - {variant:25s} : {info['description']}")
    print()
    
    if args.upload_all:
        if not args.username:
            parser.error("--upload_all requires --username")
        upload_all_models(args.username, args.num_labels)
    elif args.model_variant and args.model_name:
        upload_model_to_hub(
            model_variant=args.model_variant,
            model_name=args.model_name,
            spatial_dims=args.spatial_dims,
            num_channels=args.num_channels,
            num_labels=args.num_labels,
        )
    else:
        parser.error("--model_variant and --model_name must be specified together, or --upload_all and --username must be specified")


if __name__ == "__main__":
    main()
