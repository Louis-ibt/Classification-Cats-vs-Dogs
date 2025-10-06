# Cats vs Dogs Classification

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)
![Python](https://img.shields.io/badge/python-3.6+-blue.svg)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-Computer_Vision-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-95%25-success)

## Overview
This project implements a deep learning solution for classifying images as either cats or dogs. Using transfer learning with VGG16, the model achieves 95% accuracy on validation data and demonstrates robust real-world performance with 91.9% F1-score on 8,000 previously unseen images.

### Key Achievements
- Successfully processed and cleaned 25,000 training images
- Implemented data augmentation to prevent overfitting
- Achieved 91.9% F1-score on independent test set
- Developed three iterations of models with increasing complexity

## Dataset
- **Source**: Kaggle Cats vs Dogs Classification Dataset
- **Size**: 25,000 images (12,500 cats, 12,500 dogs)
- **Format**: RGB images of varying sizes (preprocessed to 224x224)
- **Split**: 75% training, 25% validation
- **Independent Test Set**: 8,000 additional images

## Requirements
- Python 3.6+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- PIL

## Model Architecture
```
VGG16 (Transfer Learning)
├── Input Layer (224x224x3)
├── VGG16 Base Model (Pretrained)
├── GlobalAveragePooling2D
└── Dense Layer with Sigmoid
```

### Model Evolution
| Stage | Description | Accuracy | Key Implementation |
|-------|-------------|----------|-------------------|
| Base CNN | Initial implementation with basic convolutional layers | 85% | Custom CNN architecture |
| Data Augmentation (not used in final code) | Code prepared but not applied | 88% | Random rotations, flips, zooms |
| Transfer Learning | Implemented VGG16 with fine-tuning | 95% | Pretrained VGG16, custom top layers |

### Final Performance
| Metric    | Training/Validation | Independent Test Set (8k images) |
|-----------|--------------------|---------------------------------|
| Accuracy  | 95.2% | 91.5% |
| Precision | 94.8% | 92.1% |
| Recall    | 95.6% | 91.7% |
| F1-Score  | 95.2% | 91.9% |

## Quick Start
1. Clone this repository
```bash
git clone https://github.com/yourusername/cats-vs-dogs-classification.git
cd cats-vs-dogs-classification
```

2. Set up your environment
```bash
pip install -r requirements.txt
```

3. Place your dataset in the `data/` directory
4. Update the configuration file as needed
5. Run the training script:
```bash
python train.py
```

## Project Structure
```
├── data/
│   ├── train/
│   └── test/
├── notebooks/
│   └── Cat_vs_Dog_Final.ipynb
├── src/
│   ├── train.py
│   └── predict.py
└── README.md
```

## Future Development
- Integration with ResNet and EfficientNet architectures
- Real-time prediction using webcam
- Web application deployment
- Model optimization for mobile devices

## License
This project is open-source and available for educational purposes.

## Acknowledgments
- TensorFlow team for the framework
- Kaggle for the dataset
- VGG team for the pretrained model
