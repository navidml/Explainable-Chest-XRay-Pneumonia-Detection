<div align="center">

<h1>
Explainable Chest X-Ray Pneumonia Detection using Deep Learning
</h1>

<p>
An Explainable Artificial Intelligence framework for pneumonia detection from chest X-Ray images using DenseNet121 and Grad-CAM.
</p>


<p>
<img src="https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange">
<img src="https://img.shields.io/badge/Architecture-DenseNet121-blue">
<img src="https://img.shields.io/badge/Explainability-Grad--CAM-green">
<img src="https://img.shields.io/badge/Domain-Medical%20AI-red">
</p>

</div>


---

## Overview

This project develops an **Explainable Deep Learning system** for automated pneumonia detection from chest X-ray images.

The objective is not only to classify medical images into **Normal** and **Pneumonia** categories, but also to provide visual explanations of model decisions using **Gradient-weighted Class Activation Mapping (Grad-CAM)**.

By combining transfer learning with explainable AI techniques, this system aims to improve transparency, interpretability, and reliability of deep learning models in medical imaging.


---

## Key Objectives

The main goals of this project are:

- Develop a deep learning model for pneumonia classification from chest X-ray images
- Utilize DenseNet121 transfer learning for feature extraction
- Apply medical image preprocessing and augmentation techniques
- Improve model robustness and generalization
- Evaluate performance using clinical classification metrics
- Generate visual explanations using Grad-CAM


---

## Deep Learning Approach


The model architecture follows a transfer learning strategy:


```
Chest X-Ray Image

        |
        v

Image Preprocessing

        |
        v

DenseNet121
(Pre-trained CNN Backbone)

        |
        v

Global Average Pooling

        |
        v

Fully Connected Classification Layers

        |
        v

Sigmoid Output Layer

        |
        v

Normal / Pneumonia Prediction

```


DenseNet121 was selected because of its powerful feature extraction capability and proven performance in medical image analysis.


---

## Explainable AI (Grad-CAM)


Deep learning models often behave as black-box systems, making their decisions difficult to interpret.

To address this challenge, this project integrates **Grad-CAM**, an explainability method that highlights important regions in the X-ray image that contributed to the prediction.


The explainability pipeline:


```
Input X-Ray Image

        |
        v

Deep Learning Model

        |
        v

Prediction Probability

        |
        v

Grad-CAM Heatmap

        |
        v

Visual Model Explanation

```


Grad-CAM provides insight into whether the model focuses on clinically meaningful lung regions during prediction.


---

## Dataset


The model is trained on a chest X-ray dataset containing two categories:


| Class | Description |
|------|-------------|
| Normal | Chest X-ray images without pneumonia |
| Pneumonia | Chest X-ray images showing pneumonia patterns |


The dataset is processed through:

- Image resizing
- Pixel normalization
- Data augmentation
- Class distribution analysis


---

## Technologies


### Deep Learning

- TensorFlow
- Keras
- DenseNet121
- Transfer Learning
- Fine Tuning


### Data Processing

- Python
- NumPy
- Pandas
- OpenCV
- Pillow


### Visualization

- Matplotlib
- Seaborn


### Explainable AI

- Grad-CAM


### Deployment

- Streamlit


---

## Training Pipeline


### Data Preparation

The preprocessing pipeline includes:

- Image loading
- Resizing images to model input size
- Normalization
- Data augmentation
- Dataset preparation


### Model Training

The training process includes:

- DenseNet121 initialization
- Custom classification head
- Optimized training callbacks
- Model checkpointing
- Performance monitoring


### Model Evaluation

The system is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix


---

## Model Performance


Final evaluation results:


| Metric | Score |
|--------|-------|
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1 Score | TBD |
| ROC-AUC | TBD |


Performance metrics will be updated after final model validation.


---

## Example Prediction


Example workflow:


```
Input:

Chest X-Ray Image


Model Output:

Class:
Pneumonia


Confidence:
XX %

```


The system provides both prediction results and Grad-CAM visualization for interpretation.


---

## Installation


Clone the repository:


```bash
git clone https://github.com/yourusername/Explainable-Chest-XRay-Pneumonia-Detection.git
```


Install dependencies:


```bash
pip install -r requirements.txt
```


---

## Usage


Train the model:


```bash
python src/train.py
```


Run inference:


```bash
python src/inference.py
```


Launch the Streamlit application:


```bash
streamlit run app/app.py
```


---

## Future Improvements


Future development directions include:

- Vision Transformer (ViT) based architectures
- Ensemble deep learning models
- Self-supervised medical image learning
- Large-scale clinical validation
- Deployment as an AI-assisted diagnostic support system


---

## Author


**Navid Bordbar**

Medical AI | Deep Learning | Data Science


---

## License


This project is released under the MIT License.
