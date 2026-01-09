# 🎭 Deepfake Video Detection — Ensemble Model

This project implements a **deepfake detection pipeline** that uses an **ensemble of models** to classify video frames as real or manipulated.  
The notebook includes preprocessing, feature extraction, and final decision aggregation across the full video.

---

## 🚀 Features

✔ Frame-level analysis from uploaded videos  
✔ Face extraction using OpenCV  
✔ Multiple model predictions (CNN, XGBoost, etc. based on notebook)  
✔ Ensemble voting / averaging for final decision  
✔ Easy to adapt to new datasets (FaceForensics++, DFDC, custom videos)  
✔ Fully runnable inside Kaggle or Colab

---

## 🧩 Approach

1. **Extract frames** from video  
2. **Detect faces** inside frames  
3. **Predict deepfake probability** per frame using multiple models:
   - CNN classifier (image features)
   - Secondary ML model (SVM / XGBoost / RF)
4. **Aggregate predictions** across frames
5. Output:
   - 🎯 `Real`
   - 🎭 `Fake`
   - Probability score

---

## 🧠 Model Ensemble

| Component | Role |
|----------|------|
| Convolutional Neural Network (CNN) | Learns visual artifacts from frames |
| XGBoost / SVM / RF (as chosen) | Captures non-linear patterns |
| Majority / Weighted Voting | Improves robustness |

---

## 📦 Dependencies

Install required packages:

```bash
pip install -r requirements.txt
