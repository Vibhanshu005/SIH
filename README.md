# SIH
# 🎓 Student Dropout Prediction System

This project predicts **student dropout risk** using a trained **Random Forest Classifier** and provides a **Streamlit dashboard** for interactive insights.

---

## 🚀 Features
- Predicts dropout risk (Low, Medium, High)
- Accepts student CSV data
- Interactive dashboard with:
  - 📊 Prediction results
  - 📈 Visualizations
  - 📋 Risk summaries + recommendations
- Download predictions as CSV

---

## 📂 Repository Structure
- `model/train_model.py` → trains the model, saves `.pkl` files
- `app/dropout_dashboard.py` → Streamlit dashboard app
- `requirements.txt` → project dependencies
- `sample_students.csv` → demo student dataset

---

## ⚙️ Installation
Clone the repo:
```bash
git clone https://github.com/<your-username>/student-dropout-prediction.git
cd student-dropout-prediction
