# Pneumonia Detection from Chest X-Rays using CNN and Explainable AI

This project demonstrates a deep learning approach to automatically detect pneumonia from chest X-ray images. The model is built using a Convolutional Neural Network (CNN) with transfer learning and includes an explainable AI (XAI) component using LIME to interpret the model's predictions.

![Analysis Report Example](https://i.imgur.com/39w62aB.png)
*(Example of the generated text-based analysis report for a test image)*

## Overview

Medical image analysis is a critical field where AI can significantly aid diagnostics. This project tackles the binary classification task of identifying whether a chest X-ray shows signs of pneumonia or is normal. To achieve high accuracy, it leverages the **MobileNetV2** architecture, pre-trained on ImageNet, and fine-tunes it for this specific task.

A key focus of this project is **explainability**. In medical applications, understanding *why* a model makes a certain prediction is as important as the prediction itself. This is addressed by integrating **LIME (Local Interpretable Model-agnostic Explanations)**, which highlights the specific regions in the X-ray image that the model used to make its decision.

## Features

- **Binary Classification:** Classifies chest X-rays into two categories: `NORMAL` or `PNEUMONIA`.
- **Transfer Learning:** Utilizes the powerful **MobileNetV2** architecture with pre-trained ImageNet weights for robust feature extraction.
- **Data Augmentation:** Applies random flips and rotations to the training data to improve model generalization and prevent overfitting.
- **Performance Evaluation:** Includes a detailed performance analysis with metrics like accuracy, precision, recall, F1-score, a confusion matrix, and an ROC curve.
- **Explainable AI (XAI):**
  - Implements **LIME** to visualize which parts of the X-ray image are most influential in the model's prediction.
  - Generates an **automated, human-readable text report** summarizing the model's finding, confidence level, and the evidence it used.

## Dataset

The project uses the "Chest X-Ray Images (Pneumonia)" dataset, which is sourced directly from Kaggle Hub.

- **Source:** [Kaggle Hub - Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes:** `NORMAL`, `PNEUMONIA`
- **Data Split:** The dataset is organized into `train`, `val`, and `test` directories.
- **Image Size:** All images are resized to `224x224` pixels to be compatible with the MobileNetV2 input layer.

The notebook handles the download and setup of this dataset automatically using the `kagglehub` library.

## Methodology

The project follows a standard deep learning pipeline:

1.  **Data Loading & Preprocessing:**
    - The dataset is downloaded from Kaggle Hub.
    - Images are loaded into TensorFlow datasets, resized, and normalized to a `[0, 1]` pixel value range.
    - Data augmentation is applied to the training set.

2.  **Model Architecture:**
    - A **MobileNetV2** model is used as the base, with its convolutional layers frozen (`trainable = False`).
    - A custom classification head is added on top, consisting of:
      - `GlobalAveragePooling2D`
      - `Dropout` (with a rate of 0.3)
      - `Dense` output layer with a `sigmoid` activation for binary classification.

3.  **Training:**
    - The model is compiled using the `adam` optimizer and `binary_crossentropy` loss function.
    - It is trained for 10 epochs with `EarlyStopping` and `ReduceLROnPlateau` callbacks to ensure efficient training and prevent overfitting.

4.  **Evaluation & Explainability:**
    - The trained model is evaluated on the test set to generate performance metrics.
    - **LIME** is used on sample test images to create visual explanations.
    - A custom function generates a text-based report for each prediction, making the results easy to interpret.

## Performance & Results

The model achieves strong performance on the test set, demonstrating its effectiveness in distinguishing between normal and pneumonia X-rays.

- **Accuracy:** ~87%
- **AUC (Area Under Curve):** ~0.96

### Classification Report

```
              precision    recall  f1-score   support

      NORMAL       0.93      0.71      0.80       234
   PNEUMONIA       0.85      0.97      0.90       390

    accuracy                           0.87       624
   macro avg       0.89      0.84      0.85       624
weighted avg       0.88      0.87      0.86       624
```

### Explainable AI (XAI) with LIME

LIME helps build trust in the model by showing that it focuses on clinically relevant areas (like lung opacities) when identifying pneumonia.

![LIME Explanation](https://i.imgur.com/rN4g8w8.png)
*(Example showing the original image and the LIME explanation highlighting areas contributing to the "PNEUMONIA" prediction.)*

## Technologies Used

- **Python 3**
- **TensorFlow & Keras:** For building and training the deep learning model.
- **TensorFlow Hub:** For loading the pre-trained MobileNetV2 model.
- **Scikit-learn:** For performance metrics and evaluation.
- **LIME:** For model explainability.
- **Kaggle Hub:** For automated dataset access.
- **NumPy:** For numerical operations.
- **Matplotlib & OpenCV:** For image visualization and manipulation.

## How to Run

There are two primary ways to run this project.

### 1. Using Google Colab (Recommended)

The easiest way to run this notebook is by opening it directly in Google Colab. This will provide a ready-to-use environment with a free GPU.

1.  Open the `.ipynb` file from this repository in Google Colab.
2.  Ensure the runtime is set to use a **GPU** (`Runtime` > `Change runtime type` > `T4 GPU`).
3.  Run all the cells sequentially.

### 2. Running Locally

To run the notebook on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Pneumonia-Detection-in-Chest-X-Ray-Images-Using-CNN-and-Explainable-AI.git](https://github.com/your-username/Pneumonia-Detection-in-Chest-X-Ray-Images-Using-CNN-and-Explainable-AI.git)
    cd Pneumonia-Detection-in-Chest-X-Ray-Images-Using-CNN-and-Explainable-AI
    ```

2.  **Set up a Python environment:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    A `requirements.txt` file can be created with the following content:
    ```
    tensorflow
    tensorflow-hub
    scikit-learn
    lime
    kagglehub
    matplotlib
    opencv-python
    ```
    Then install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook pneumonia_detection_in_check_xray_final_code.ipynb
    ```

5.  Run the cells in the notebook.
