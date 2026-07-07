# 🩺 Explainable Chest X-Ray Pneumonia Detection using Deep Learning

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Model-DenseNet121-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Explainability-Grad--CAM-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Domain-Medical%20AI-red?style=for-the-badge">
</p>


## 📌 Overview

This project presents an **Explainable Artificial Intelligence (XAI) system** for detecting pneumonia from chest X-ray images using advanced deep learning techniques.

The goal is not only to classify X-ray images as **Normal** or **Pneumonia**, but also to provide visual explanations of model decisions using **Grad-CAM**, improving transparency and interpretability for medical applications.

The system uses **DenseNet121 Transfer Learning**, optimized training strategies, and explainable AI methods to create a reliable medical imaging pipeline.


---

# 🎯 Project Objectives

The main objectives of this project are:

✅ Develop a deep learning model for pneumonia detection from chest X-ray images

✅ Apply transfer learning using DenseNet121 architecture

✅ Handle medical image classification challenges

✅ Improve model generalization through data augmentation

✅ Evaluate performance using appropriate classification metrics

✅ Visualize model attention using Grad-CAM explanations


---

# 🧠 Model Architecture


```
Chest X-Ray Image

        │

        ▼

Image Preprocessing

        │

        ▼

DenseNet121
(Transfer Learning Backbone)

        │

        ▼

Global Average Pooling

        │

        ▼

Fully Connected Layers

        │

        ▼

Sigmoid Classifier

        │

        ▼

Normal / Pneumonia Prediction

```


---

# 🔬 Explainable AI with Grad-CAM

Deep learning models are often considered black boxes.

To overcome this limitation, this project integrates **Gradient-weighted Class Activation Mapping (Grad-CAM)**.

Grad-CAM highlights the regions of the X-ray image that influenced the model prediction.


Example workflow:

```
X-Ray Image

      ↓

Deep Learning Model

      ↓

Prediction

      ↓

Grad-CAM Heatmap

      ↓

Visual Explanation

```


This allows better understanding of model behavior and increases trust in AI-assisted medical diagnosis.


---

# 📂 Dataset

The project uses a chest X-ray pneumonia dataset containing two classes:

| Class | Description |
|---|---|
| Normal | Healthy chest X-ray images |
| Pneumonia | X-ray images showing pneumonia patterns |


Dataset structure:

```
chest_xray/

│

├── train/

│   ├── NORMAL/

│   └── PNEUMONIA/


├── validation/

│   ├── NORMAL/

│   └── PNEUMONIA/


└── test/

    ├── NORMAL/

    └── PNEUMONIA/

```


---

# ⚙️ Technologies Used


## Deep Learning

- TensorFlow
- Keras
- DenseNet121
- Transfer Learning
- Fine Tuning


## Data Processing

- NumPy
- Pandas
- OpenCV
- Pillow


## Visualization

- Matplotlib
- Seaborn


## Explainable AI

- Grad-CAM


## Deployment

- Streamlit


---

# 🚀 Training Pipeline


The complete pipeline includes:


### 1. Data Preparation

- Image resizing
- Normalization
- Data augmentation
- Class distribution analysis


### 2. Model Development

- Loading DenseNet121 pretrained weights
- Adding custom classification layers
- Training with optimized callbacks


### 3. Model Evaluation

Evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix


### 4. Explainability

Generation of Grad-CAM heatmaps for model interpretation.


---

# 📊 Results


Final model performance:


| Metric | Score |
|---|---|
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1 Score | TBD |
| ROC-AUC | TBD |


*(Results will be updated after final training)*


---

# 🖼️ Example Predictions


## Model Prediction

```
Input:
Chest X-Ray Image


Output:

Prediction: Pneumonia

Confidence: XX%

```


## Grad-CAM Explanation

The model highlights important lung regions contributing to the prediction.


---

# 🛠️ Installation


Clone repository:


```bash
git clone https://github.com/yourusername/Explainable-Chest-XRay-Pneumonia-Detection.git
```


Navigate into project:


```bash
cd Explainable-Chest-XRay-Pneumonia-Detection
```


Install dependencies:


```bash
pip install -r requirements.txt
```


---

# ▶️ Usage


## Training


```bash
python src/train.py
```


## Prediction


```bash
python src/inference.py
```


## Streamlit Application


```bash
streamlit run app/app.py
```



---

# 🌟 Future Improvements


Possible improvements:

- Vision Transformer (ViT) implementation
- Ensemble deep learning models
- Self-supervised pretraining
- Deployment as medical decision support system
- External clinical validation


---

# 👨‍💻 Author


**Navid Bordbar**

Medical AI | Deep Learning | Data Science


---

# ⭐ Acknowledgements


This project demonstrates the potential of Artificial Intelligence in medical imaging and aims to improve transparency and reliability of deep learning-based diagnostic systems.


---

# 📜 License

This project is licensed under the MIT License.
