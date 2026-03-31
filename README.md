# 🥗 AI Personal Diet Consultant

![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![Framework](https://img.shields.io/badge/framework-Streamlit-red)
![Model](https://img.shields.io/badge/model-Random%20Forest-green)
![Accuracy](https://img.shields.io/badge/accuracy-93%25-brightgreen)

An interactive Web Application that uses a machine learning classifier to recommend personalized diet plans based on a user's unique biological metrics and metabolic markers.

## ✨ Features

-   **Interactive UI:** A modern, responsive interface built with Streamlit and custom CSS.
-   **Real-time Prediction:** Uses a Random Forest model to instantly classify the best diet type.
-   **Leakage-Proof Model:** The model has been audited to remove data leakage (dropped BMI and Goal features) for realistic real-world performance.
-   **Balanced Learning:** Trained on data balanced via SMOTE (Synthetic Minority Over-sampling Technique) to ensure fairness for all diet categories.

---

## 📸 Screenshots

### 1. Application Interface
Users enter their Age, Gender, Weight, Height, Activity Level, Sugar, and Cholesterol.

<img width="1919" height="949" alt="image" src="https://github.com/user-attachments/assets/13eb73ee-732f-45e1-af3d-9b94e4646e1a" />


### 2. Personalized Recommendation
After clicking the button, the AI analyzes the metabolic profile and generates a visual recommendation card.

<img width="1906" height="949" alt="image" src="https://github.com/user-attachments/assets/d64603ba-5ff7-4d02-af26-15e67c7a14d3" />


---

## 🧠 Model Methodology

### The Balancing Act (SMOTE)
The original dataset was heavily imbalanced (e.g., Diabetic class had 500+ samples while Balanced had only ~40). To prevent model bias, **SMOTE** was used to synthetically oversample the minority classes, ensuring the final model saw 539 examples of *every* diet type.

<img width="1919" height="956" alt="image" src="https://github.com/user-attachments/assets/7713cb9a-ffcf-4fdf-842e-8381008f66c3" />


### Performance Evaluation
The final Random Forest model achieved a **93.1% accuracy** on the test set. The confusion matrix below shows robust performance across all categories, proving the success of the SMOTE implementation.

<img width="1917" height="935" alt="image" src="https://github.com/user-attachments/assets/e42c0c9f-25f8-4e0b-8718-67a69e6e4775" />


---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.11 or 3.12 (A stable version is required).
* Git

### Step-by-Step

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Diet_Prediction_AI.git](https://github.com/YOUR_USERNAME/Diet_Prediction_AI.git)
    cd Diet_Prediction_AI
    ```

2.  **Create and Activate a Virtual Environment:**
    ```powershell
    # Windows (PowerShell)
    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

3.  **Install Required Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

---

## 📦 File Structure

```text
Diet_Prediction_AI/
│
├── .venv/                 # Virtual environment (ignored by Git)
├── screenshots/           # Images used in README
│   ├── app_interface.png
│   ├── confusion_matrix.png
│   └── ...
├── app.py                 # Final Streamlit application
├── Final_Diet_Model.pkl   # Saved balanced Random Forest model
├── Final_Scaler.pkl       # Saved 7-feature StandardScaler
├── Diet_Labels.pkl        # Saved Label Encoder classes
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
