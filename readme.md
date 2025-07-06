# Fake Image & AI-Generated Image Detector

## Overview
The **Fake Image & AI-Generated Image Detector** is a machine learning-based web application designed to identify whether an image is real, fake, or AI-generated. It uses **Error Level Analysis (ELA)** and **Photo Response Non-Uniformity (PRNU)** noise features to detect manipulation and synthetic content. The system supports two separate models:
- A CNN-based model for detecting fake image manipulations.
- A dual-input CNN model for detecting AI-generated images using ELA + PRNU features.

The app also extracts and displays metadata from the uploaded images for further context.

## Features
- **Error Level Analysis (ELA):** Highlights discrepancies caused by image manipulation.
- **PRNU Feature Extraction:** Analyzes sensor noise patterns to detect AI-generated content.
- **Metadata Extraction:** Displays software used, camera information, dimensions, and other relevant metadata.
- **Real-Time Prediction:** Classifies images as *Real*, *Fake*, or *AI Generated* with confidence scores.
- **Interactive Interface:** User-friendly interface built with **Streamlit** or **Flask** for seamless interaction.

## Dataset
- The **Fake Image Detection** model is trained on the **CASIA2** dataset.
- The **AI-Generated Image Detection** model uses a custom-labeled dataset containing real and AI-generated images (e.g., from ThisPersonDoesNotExist).
- All datasets are preprocessed using **ELA**; AI detection additionally uses **PRNU noise extraction**.
- Fake Image dataset - https://mega.nz/file/m4AF0IoZ#wjQ5aSEE1It-l-QebtKpsQMDGeWCGIdqqvNrRe5dbvc
- Ai generated image dataset - https://mega.nz/file/Xg5wRIKa#Jt4M7RSTavzWA66llsl8SlkqtBC9NEP1szkvKGuGnoY

## Workflow

### Dataset Preprocessing
- **ELA Conversion:** Converts images to ELA format to detect manipulation or synthetic patterns.
- **PRNU Extraction:** Extracts sensor-based noise patterns from images for AI-detection.
- **Normalization & Reshaping:** Normalizes and reshapes image data for model input.

### CNN Model (Fake Image Detection)
- **Architecture:** A sequential model with **Conv2D**, **MaxPool2D**, **Dense**, and **Dropout** layers.
- **Classification:** Performs binary classification using the **softmax** activation function.

### Dual-Input CNN Model (AI Image Detection)
- **Architecture:** Two separate CNN branches for ELA and PRNU features.
- **Merging:** ELA and PRNU features are concatenated and passed through Dense layers.
- **Classification:** Performs binary classification with **sigmoid** activation.

### Training
- **Compilation:** Uses `binary_crossentropy` loss function and `accuracy` metric.
- **Early Stopping:** Implements early stopping to avoid overfitting during training.
- Both models are trained separately using their respective datasets.

### Model Loading
- Loads the pre-trained CNN models:
  - `model_casia.h5` for fake image detection
  - `model_ai.h5` for AI-generated image detection

### Model Architecture

#### Fake Image Detection
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

#### AI-Generated Image Detection (Dual-Input)
- **Input Shapes:** 
  - ELA: (128, 128, 3)
  - PRNU: (128, 128, 1)
- **ELA Branch:** Conv2D → MaxPooling → Dropout → Flatten
- **PRNU Branch:** Conv2D → MaxPooling → Dropout → Flatten
- **Concatenation:** Merge both branches
- **Dense Layers:** 256 → 128 → 1 (sigmoid activation)

## Technologies Used
- **Python Libraries:**
  - TensorFlow/Keras: For model building and training
  - Streamlit / Flask: For creating the web application
  - PIL: For image processing
  - NumPy: For array manipulation
- **Techniques:**
  - **Error Level Analysis (ELA):** For detecting image manipulation
  - **PRNU Extraction:** For identifying AI-generated images

---

## Contributors
- **Anish Das**
- **Arpan Halder**
- **Ajoy Mondal**
- **Laxmi Singh**
- **Ankita Chowdhury**
