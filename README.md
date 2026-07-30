# 🖼️ AI Vision Studio – Image Recognition using MobileNetV2

An AI-powered web application that recognizes everyday objects from images using **Computer Vision**, **Deep Learning**, and **Transfer Learning**.

Built with **Python**, **TensorFlow**, **MobileNetV2**, and **Streamlit**.

---

## 📌 Project Overview

This project uses a pretrained **MobileNetV2** deep learning model to identify objects from uploaded images.

Instead of training a model from scratch, it leverages **Transfer Learning**, where MobileNetV2 has already been trained on the **ImageNet** dataset containing over **1 million images** across **1000 object categories**.

The application allows users to upload an image, processes it through the neural network, and displays the **Top-5 predicted objects** along with their confidence scores.

---

## 🚀 Features

- 🖼️ Upload JPG, JPEG, or PNG images
- 🤖 AI-powered object recognition using MobileNetV2
- 🥇 Displays Top-5 predictions
- 📊 Confidence score visualization
- ⚡ Real-time prediction with inference time
- 🎯 Interactive AI Challenge Zone
- 📚 Step-by-step explanation of the prediction workflow
- 🎓 Interview preparation section
- 📱 User-friendly Streamlit interface

---

## 🛠️ Technologies Used

- Python
- Streamlit
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Pandas
- Pillow
- Matplotlib
- Seaborn

---

## 📂 Project Structure

```text
Image-Recognition/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── Notebook/
│   └── Image_Recognition.ipynb
│
├── Images/
│   ├── cat.jpg
│   ├── dog.jpg
│   ├── bird.jpg
│   ├── bottle.jpg
│   ├── car.jpg
│   └── food.jpg
│
└── screenshots/
    ├── homepage.png
    └── prediction.png
```

---

## 🧠 Deep Learning Workflow

```text
Input Image
      │
      ▼
Resize to 224 × 224
      │
      ▼
Image Preprocessing
      │
      ▼
MobileNetV2 Neural Network
      │
      ▼
Class Probabilities
      │
      ▼
Top-5 Predictions
```

---

## 📸 Application Preview

### 🏠 Home Page

![Home Page](screenshots/homepage.png)

---

### 🎯 Prediction Result

![Prediction Result](screenshots/prediction.png)

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/your-username/Image-Recognition.git
```

### Navigate to the project

```bash
cd Image-Recognition
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## 📊 Model Information

| Model | MobileNetV2 |
|-------|-------------|
| Framework | TensorFlow / Keras |
| Dataset | ImageNet |
| Classes | 1000 |
| Technique | Transfer Learning |
| Input Image Size | 224 × 224 |

---

## 🎯 Student Challenge

Try testing the model with different types of images:

- 🐶 Dog
- 🐱 Cat
- 🚗 Car
- 🐦 Bird
- 🍔 Food
- 🍼 Bottle

### Experiment with:

- Blurry images
- Dark images
- Multiple objects
- Rotated images
- Close-up images

Observe how confidence scores and predictions change.

---

## 💡 Key Learning Outcomes

- Understanding Computer Vision fundamentals
- Applying Transfer Learning
- Using pretrained deep learning models
- Image preprocessing techniques
- Building AI-powered web applications with Streamlit
- Interpreting confidence scores and prediction probabilities

---

## 🎯 Future Improvements

- Support custom-trained image classification models
- Webcam-based real-time object recognition
- Object detection with bounding boxes (YOLO)
- Multi-language interface
- Prediction history
- Image segmentation
- Deploy on cloud platforms

---

## 👩‍💻 Author

** Sonali **

B.Tech Computer Science (Data Science)

---

## ⭐ If you like this project

Give it a ⭐ on GitHub.