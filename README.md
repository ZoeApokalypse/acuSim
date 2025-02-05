# AcuSim: A Synthetic Dataset for Cervicocranial Acupuncture Points Localisation

[![CC BY 4.0][cc-by-shield]][cc-by]

This dataset is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

## Dataset Overview
A multi-view acupuncture point dataset containing:
- 64x64, 128x128, 256x256, 512×512 and 1024x1024resolution RGB images
- Corresponding JSON annotations with:
  - 2D/3D keypoint coordinates
  - Visibility weights (0.9-1.0 scale)
  - Meridian category indices
  - Visibility masks
- 174 standard acupuncture points (map.txt)
- Occlusion handling implementation

## Key Features
- **Multi-view Rendering**: Generated using Blender 3.5 with realistic occlusion simulation
- **Structured Annotation**:
  - Default initialization for occluded points ([0.0, 0.0, 0.5])
  - Meridian category preservation for occluded points
  - Weighted visibility scoring
- **ML-Ready Format**: Preconfigured PyTorch DataLoader implementation

## Dataset Structure
```
dataset_root/
├── map.txt                 # Complete list of 174 acupuncture points
├── train/
│   ├── image/img_512/      # Training images (*.png)
│   └── label/label/        # JSON annotations (*.json)
└── test/                   # [Optional] Similar structure for testing
```

## Quick Start
### Dependencies
```python
Python 3.10
PyTorch 2.0+
TensorFlow 2.60 (GPU recommended)
CUDA 12.4
cuDNN 9.5.0
blender 3.6.n LTS under python 3.10
```

### Basic Usage
```python
from acuSim_dataloader_modified import AcuPointsDataset, create_category_encoding

# Initialize dataset
dataset = AcuPointsDataset(
    image_dir="path/to/images",
    json_dir="path/to/labels",
    target_size=(512, 512),
    transform=transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
)

# Create DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

## Advanced Features
### Key Components
**Occlusion Handling**:
   - Preserves meridian category for occluded points
   - Automatic mask generation (`create_mask()`)
   - Weighted visibility scores in JSON annotations


## Code Availability
| Script | Purpose | Requirements |
|--------|---------|--------------|
| [`script.py`](https://github.com/ZoeApokalypse/acuSim) | Blender rendering pipeline | Blender 3.5 |
| [`xuewei5.py`](https://github.com/ZoeApokalypse/acuSim) | Visualization examples | Python 3.10 |
| [`dataloader_modified.py`](https://github.com/ZoeApokalypse/acuSim) | PyTorch/TF data loading | TF 2.60+ |
| [`occlusion_detection.py`](https://github.com/ZoeApokalypse/acuSim) | Occlusion generation | CUDA 12.4 |

## Citation
Please cite our work when using this dataset:
```bibtex
@article{
}
```

## Acknowledgments
Supported by:
- XJTLU AI University Research Centre
- Jiangsu Province Engineering Research Centre of Data Science and Cognitive Computation
- Suzhou Municipal Key Laboratory for Intelligent Virtual Engineering (SZS2022004)

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg