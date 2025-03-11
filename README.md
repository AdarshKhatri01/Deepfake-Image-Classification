# **🚀 Deepfake Image Detection Using CNNs**  
AI-powered deepfake detection using deep learning models like ResNet50, VGG16, DenseNet, and a custom CNN.  

---

## **📌 Project Overview**
Deepfake images are AI-generated fake images that pose security and misinformation threats. This project implements **deep learning-based models** to classify images as **real or fake** using **Convolutional Neural Networks (CNNs)**. It compares multiple pretrained models along with a **custom CNN architecture** to achieve optimal detection performance.

---

## **📌 Features**
✅ **Multi-Model Approach**: Uses **ResNet50, VGG16, VGG19, DenseNet121, DenseNet169, DenseNet201, and Custom CNN**.  
✅ **Dataset**: 140K Real and Fake Faces Dataset (70K real + 70K fake images).  
✅ **Training & Evaluation**: Compares models based on **accuracy, precision, recall, F1-score, and confusion matrix**.  
✅ **Web Application (on the way)**: Flask-based **web app** for real-time image classification.   
✅ **Model Saving & Reloading**: Saves trained models and history for later analysis.  
✅ **Performance Graphs**: Plots training & validation accuracy/loss.  

---

## **📌 Dataset Details**
📌 **Dataset Name**: 140K Real and Fake Faces  
📌 **Source**: [Kaggle Dataset](https://www.kaggle.com/xhlulu/140k-real-and-fake-faces)  
📌 **Distribution**:  
- **Train Set**: 100,000 images (50K real + 50K fake)  
- **Validation Set**: 20,000 images (10K real + 10K fake)  
- **Test Set**: 20,000 images (10K real + 10K fake)  

### **📂 Dataset Structure**
The dataset is organized in the following structure:

```
/dataset
│── /train
│   │── /real
│   │   ├── real_1.jpg
│   │   ├── real_2.jpg
│   │   ├── ...
│   │── /fake
│       ├── fake_1.jpg
│       ├── fake_2.jpg
│       ├── ...
│── /valid
│   │── /real
│   │── /fake
│── /test
    │── /real
    │── /fake
```

Each subfolder contains images categorized as **real or fake**.

---

## **📌 Model Architectures Used**
1️⃣ **Custom CNN** (6 Convolutional layers + BatchNorm, MaxPooling, Dropout)  
2️⃣ **Pretrained Models** (Feature Extraction & Fine-Tuning):
   - ResNet50  
   - VGG16  
   - VGG19  
   - DenseNet121  
   - DenseNet169  
   - DenseNet201  

---

## **📌 Installation & Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/AdarshKhatri01/Deepfake-Image-Classification.git
cd Deepfake-Image-Classification
```

### **2️⃣ Install Required Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Download Dataset**
- Download the dataset from Kaggle and extract it into the `dataset/` folder.

---

## **📌 Model Evaluation Results**
| Model       | Accuracy | Precision | Recall | F1-Score |
|------------|----------|------------|--------|----------|
| **Custom CNN** | **0.98** | **0.98** | **0.98** | **0.98** |
| **DenseNet121** | 0.93 | 0.93 | 0.93 | 0.93 |
| **DenseNet201** | 0.93 | 0.93 | 0.93 | 0.93 |
| **DenseNet169** | 0.93 | 0.93 | 0.93 | 0.93 |
| **ResNet50** | 0.92 | 0.92 | 0.92 | 0.92 |
| **VGG16** | 0.90 | 0.91 | 0.90 | 0.90 |
| **VGG19** | 0.88 | 0.88 | 0.88 | 0.88 |

📌 **Best Model:** ✅ **Custom CNN (98% Accuracy)**  
---

## **📌 Future Improvements**
🔹 **Hyperparameter Tuning** (Batch size, Learning Rate, Dropout)  
🔹 **Data Augmentation** to improve generalization  
🔹 **Deploy Model on Cloud** (AWS, Heroku, or Google Cloud)  
🔹 **Improve Speed for Real-Time Detection**  
🔹 **Expand Dataset for More Robust Detection**  

---

## **📌 Contributors**
👤 **Name** - *Adarsh Khatri*  
👤 **Supervisor’s** - *Dr. Shatrughan Modi*  

---

## **📌 License**
This project is open-source under the **MIT License**.

---

## **📌 Star ⭐ the Repository if You Like It!**
If you found this project helpful, please consider **starring** ⭐ the repository to support future improvements! 🚀🔥  

---
