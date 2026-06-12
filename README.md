# Deep Learning Image Classifier: Cats vs. Dogs

(Dataset Link)
(https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)


A robust, end-to-end Image Classification pipeline built from scratch using **PyTorch** to classify images of cats and dogs[cite: 1]. The architecture utilizes a custom **Convolutional Neural Network (CNN)** optimized with regularization techniques to handle a balanced dataset of 5,000 images[cite: 1].

---

## 📌 Project Overview
This repository contains the complete implementation of a binary image classifier[cite: 1]. The primary focus of this project was to transition from foundational raw NumPy matrix implementations (such as building MNIST digit recognizers) into a modular, production-ready **PyTorch** framework[cite: 1]. 

The pipeline includes a custom CNN architecture, comprehensive data preprocessing, robust regularization to combat overfitting, and a structured evaluation pipeline[cite: 1].

---

## 📊 Dataset Specifications
* **Total Images:** 5,000 balanced images[cite: 1]
* **Classes:** 2 (`Cat` and `Dog`)[cite: 1]
* **Distribution:** 2,500 images per class[cite: 1]
* **Data Splits:** 
  * **Training Set:** 80% (for model parameter optimization)[cite: 1]
  * **Validation Set:** 20% (for hyperparameter tuning and early stopping metrics)[cite: 1]

---

## 🏗️ Model Architecture
The network is structured as a deep Convolutional Neural Network designed to extract spatial hierarchies from raw images[cite: 1]. It sequentially applies feature extraction layers followed by a fully connected classification head[cite: 1].

| Layer Type | Configuration Details | Purpose |
| :--- | :--- | :--- |
| **Input Layer** | 3-Channel RGB Images resized to $3 \times 224 \times 224$ | Standardizes input dimensional spatial matrix[cite: 1] |
| **Convolutional Blocks** | 2D Convolutions ($3 \times 3$ Kernels, Stride=1, Padding=1) | Extracts spatial features & localized patterns[cite: 1] |
| **Batch Normalization** | `nn.BatchNorm2d` after each Conv layer | Stabilizes training dynamics, mitigates internal covariate shift[cite: 1] |
| **Activation Layer** | Rectified Linear Unit (`ReLU`) | Introduces non-linearity to the network[cite: 1] |
| **Pooling Layer** | Max Pooling ($2 \times 2$ Kernel, Stride=2) | Downsamples spatial dimensions, provides translation invariance[cite: 1] |
| **Regularization Layer**| Dropout (`p=0.3` to `0.5`) in Dense layers | Prevents co-adaptation of features, reduces overfitting[cite: 1] |
| **Fully Connected** | Linear Layers (`nn.Linear`) mapped to 2 output nodes | Classifies extracted feature embeddings into target logits[cite: 1] |

---

## ⚙️ Data Pipeline & Augmentation
To maximize structural generalization and prevent the model from memorizing geometric configurations, input images undergo a strict preprocessing pipeline via `torchvision.transforms`[cite: 1]:

1. **Resizing & Cropping:** Images are resized to $256 \times 256$ and center-cropped to $224 \times 224$ to ensure dimension uniformity across minibatches[cite: 1].
2. **Data Augmentation (Training Only):**
   * **Random Horizontal Flips:** Introduces horizontal reflection symmetry invariance[cite: 1].
   * **Random Rotation ($\pm15^\circ$):** Accounts for slight structural orientation variances[cite: 1].
   * **Color Jitter:** Minor adjustments to brightness and contrast to robustify the model against lighting alterations[cite: 1].
3. **Normalization:** Pixel values are scaled to $[0, 1]$ and normalized using ImageNet Channel Means and Standard Deviations[cite: 1]:
   $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$[cite: 1]

---

## 🚀 Technical Implementation Details
* **Framework:** PyTorch[cite: 1]
* **Loss Function:** `nn.CrossEntropyLoss()` (Binary Classification Logits)[cite: 1]
* **Optimizer:** Adam Optimizer with decoupled weight decay or standard Adam (`lr=0.001`)[cite: 1]
* **Batch Size:** 32 or 64 (Optimized for GPU memory allocation)[cite: 1]
* **Symmetry Breaking:** All weights are initialized using Kaiming (He) Normal Initialization to ensure effective gradient propagation during initial backpropagation steps[cite: 1].

---

## 📂 Repository Structure
```text
├── data/
│   ├── train/              # 80% of dataset sorted by class folders
│   └── val/                # 20% of dataset sorted by class folders
├── src/
│   ├── model.py            # PyTorch CNN Architecture definition
│   ├── dataset.py          # Custom PyTorch Dataset class & augmentation pipeline
│   ├── train.py            # Training loop script (Forward/Backward passes, tracking)
│   └── evaluate.py         # Metrics extraction script (Accuracy, Loss, Confusion Matrix)
├── notebooks/
│   └── exploration.ipynb   # Initial prototyping and dataset visualization
├── requirements.txt        # Python dependency manifest
└── README.md               # Project documentation
```[cite: 1]

---

## 📈 Execution and Workflow

### 1. Installation & Environment Setup
Clone the repository and install the verified dependencies[cite: 1]:
```bash
git clone [https://github.com/yourusername/cat-dog-classifier.git](https://github.com/yourusername/cat-dog-classifier.git)
cd cat-dog-classifier
pip install -r requirements.txt
```[cite: 1]

### 2. Training the Model
Run the primary training pipeline execution script[cite: 1]. The script handles forward propagation, loss calculation, backpropagation of gradients, and validation tracking per epoch[cite: 1]:
```bash
python src/train.py --epochs 25 --batch_size 32 --lr 0.001
```[cite: 1]

### 3. Running Evaluation
To evaluate the final weights file against the validation tracking subset and export classification performance metrics[cite: 1]:
```bash
python src/evaluate.py --model_path saved_models/best_model.pth
```[cite: 1]

---

## 🎯 Key Observations & Learning Outtakes
* **Symmetry Breaking Matters:** Manual zero-initialization locks gradients; applying explicit Kaiming Normal initialization allows the network to begin learning discriminative features immediately from Epoch 1[cite: 1].
* **Regularization Impact:** Due to the concise dataset constraints (5,000 images total), the addition of **Batch Normalization** combined with **Dropout** and random geometric transforms successfully prevented early-stage training divergence, closing the gap between training and validation loss curves[cite: 1].
* **Transitioning from NumPy to PyTorch:** Implementing the backpropagation calculus by hand in prior projects (like MNIST) made utilizing PyTorch’s `autograd` and built-in architectural layers incredibly intuitive, emphasizing how tracking tensor shapes across multi-dimensional operations controls the underlying network dynamics[cite: 1].
