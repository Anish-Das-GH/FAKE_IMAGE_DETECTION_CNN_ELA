# Fake Image Detector

## Overview
The **Fake Image Detector** is a machine learning-based web application designed to identify whether an image is real or fake using **Error Level Analysis (ELA)**. It preprocesses uploaded images, applies ELA, and predicts their authenticity with a pre-trained **Convolutional Neural Network (CNN)** model. Additionally, the app extracts and displays metadata from the uploaded images.

## Features
- **Error Level Analysis (ELA):** Highlights discrepancies caused by image manipulation.
- **Metadata Extraction:** Displays software used, camera information, dimensions, and other relevant metadata.
- **Real-Time Prediction:** Classifies images as *Real* or *Fake* with confidence scores.
- **Interactive Interface:** User-friendly interface built with **Streamlit** for seamless interaction.

## Dataset
The model is trained on the **CASIA2** dataset, which contains both real and fake images. The dataset is preprocessed using **ELA** to emphasize differences that indicate manipulation.

## Workflow

### Dataset Preprocessing
- **ELA Conversion:** Converts images to ELA format to detect manipulation.
- **Normalization & Reshaping:** Normalizes and reshapes image data for model input.

### CNN Model
- **Architecture:** A sequential model with **Conv2D**, **MaxPool2D**, **Dense**, and **Dropout** layers.
- **Classification:** Performs binary classification using the **softmax** activation function.

### Training
- **Compilation:** Uses `binary_crossentropy` loss function and `accuracy` metric.
- **Early Stopping:** Implements early stopping to avoid overfitting during training.

### Model Loading
- Loads the pre-trained CNN model (`model_casia_run.h5`) for making predictions.

### Model Architecture
- **Input Shape:** (128, 128, 3)
- **Layers:**
  - **Conv2D:** 32 filters, (5x5) kernel, ReLU activation
  - **Conv2D:** 32 filters, (5x5) kernel, ReLU activation
  - **MaxPooling2D:** (2x2) pool size
  - **Dropout:** 25%
  - **Flatten**
  - **Dense:** 256 units, ReLU activation
  - **Dropout:** 50%
  - **Dense:** 2 units, softmax activation

## Technologies Used
- **Python Libraries:**
  - TensorFlow/Keras: For model building and training
  - Streamlit: For creating the web application
  - PIL: For image processing
  - NumPy: For array manipulation
- **Techniques:**
  - **Error Level Analysis (ELA):** For detecting image manipulation