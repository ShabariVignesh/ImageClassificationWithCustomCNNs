# ImageClassificationWithCustomCNNs

This repository contains a Jupyter Notebook for classifying images of animals (cats, dogs, and foxes) using two different approaches:
1. A custom Convolutional Neural Network (CNN) that is trained from scratch.
2. A pre-trained GoogleNet model that is fine-tuned for the task.

The dataset is created using the Flickr API, and this notebook showcases key processes like data preprocessing, training both models, and comparing the results using different metrics.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Overview](#model-overview)
- [Results](#results)
- [License](#license)

## Dataset

The dataset contains 200 images per category: 
- Cats
- Dogs
- Foxes

The images are downloaded using the free Flickr API. The dataset is split into:
- **Training set** (80%)
- **Test set** (20%)

## Installation

To run the notebook, ensure that you have the following libraries installed:

```bash
pip install torch torchvision scikit-learn
```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/animal-image-classification.git
    ```
2. Open the `ShabariVignesh_Assignment3.ipynb` notebook in JupyterLab or Jupyter Notebook.
3. Run the cells to preprocess the data, split it into training and testing sets, and implement both models (Custom CNN and Pre-trained GoogleNet).

## Model Overview

### 1. Custom CNN
- **Architecture**: A simple CNN with 2 convolutional layers and 1 fully connected layer with 128 units and dropout.
- **Training**: The CNN is trained from scratch on the dataset using standard backpropagation and stochastic gradient descent.
- **Evaluation Metrics**: 
  - Initial Test Accuracy: **50.41%** after epoch 1.
  - Best Test Accuracy: **61.98%** after 20 epochs.
  - Final Training Loss: **0.3892** after 20 epochs.

### 2. Pre-trained GoogleNet
- **Architecture**: The GoogleNet model is a pre-trained model from PyTorch's model zoo. It is fine-tuned on the animal classification task.
- **Training**: Only the last layers of GoogleNet are retrained for this task, leveraging its pre-trained weights.
- **Evaluation Metrics**:
  - Initial Test Accuracy: **74.38%** after epoch 1.
  - Best Test Accuracy: **92.56%** after 17 epochs.
  - Final Training Loss: **0.2668** after 20 epochs.

## Results

The results for both models differ based on the architecture and training strategy:

- **Custom CNN**:
  - **Best Test Accuracy**: 61.98%
  - **Final Training Loss**: 0.3892
  - Slower improvement in test accuracy with some fluctuations.

- **Pre-trained GoogleNet**:
  - **Best Test Accuracy**: 92.56%
  - **Final Training Loss**: 0.2668
  - Quickly converged due to pretrained layers, with stable and high performance.

## License

This project is licensed under the MIT License.
