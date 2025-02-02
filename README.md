# ResNet50-AirplaneClassifier

# Fine-Grained Airplane Classification using ResNet-50

This project focuses on fine-grained classification of **airplane models** using deep learning. A **ResNet-50** model with transfer learning is employed to achieve high accuracy in distinguishing airplane types from images.

## Project Overview
Fine-grained classification is a challenging task where models need to distinguish subtle differences between classes. This project applies deep learning techniques to classify airplane models accurately. The dataset, sourced from **Kaggle**, consists of images of various airplane types. **ResNet-50** is used with transfer learning to leverage pre-trained ImageNet weights for better generalization.

## Objectives
- **Develop a fine-grained classification model** for airplane images.
- **Leverage transfer learning** using ResNet-50.
- **Optimize hyperparameters** to improve classification accuracy.

## Key Goals
- Implement a **PyTorch-based deep learning pipeline** for airplane classification.
- Use **data augmentation and normalization** to improve model robustness.
- Fine-tune **ResNet-50** for improved performance on the dataset.

## Tools & Technologies
- **Python**: Core programming language.
- **PyTorch & torchvision**: Deep learning framework for model training.
- **ResNet-50**: Pre-trained model for transfer learning.
- **Matplotlib & Seaborn**: Visualization tools for analyzing performance.
- **Pandas**: Data handling and preprocessing.

## Workflow Highlights
1. **Dataset Preprocessing**:
   - Images resized to **512x512**.
   - Normalized using **ImageNet mean & std**.
   - Data augmentation applied for generalization.
2. **Model Selection & Training**:
   - Used **ResNet-50 with transfer learning**.
   - Fine-tuned layers for airplane classification.
   - Optimized using **Adam optimizer** with learning rate **0.001**.
3. **Evaluation**:
   - Model tested on **test dataset**.
   - Performance assessed using accuracy and loss metrics.

## Insights
- **Transfer learning significantly improves model performance** by leveraging ImageNet weights.
- **Fine-grained classification requires high-resolution images and specialized augmentation techniques.**
- **Batch normalization and adaptive learning rates** help prevent overfitting.

## Dataset Details
- **Source**: Kaggle airplane dataset.
- **Classes**: Multiple airplane models.
- **Training & Testing Split**: Data is divided into training, validation, and test sets.

## Getting Started
### Prerequisites
- Python 3.x
- PyTorch
- torchvision, pandas, matplotlib, seaborn
- Kaggle dataset (download and extract)

## Results

- **Achieved high accuracy** in airplane classification.
- **Fine-tuned ResNet-50 model** outperforms baseline approaches.
- **Generated confusion matrix & accuracy plots** for performance evaluation.
