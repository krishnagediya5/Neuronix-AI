# 🧠 Neuronix AI : Enterprise-Grade AutoML Workspace

<div align="center">
  <img src="https://img.shields.io/badge/Status-Live-success?style=for-the-badge&logo=vercel" alt="Status" />
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" alt="Python Version" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn" alt="Machine Learning" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</div>

<br>

**Think Smarter. Predict Faster.**  
**Neuronix AI** is a state-of-the-art, end-to-end Automated Machine Learning (AutoML) web platform. It bridges the gap between raw data and actionable AI insights by allowing users to seamlessly profile data, engineer features, train high-performance models, and generate comprehensive PDF reports—all without writing a single line of code.

🚀 **Experience the Live Application:** [Neuronix AI Live Demo](https://neuronix-krishna.streamlit.app) *(Replace with your actual Streamlit link)*

---

## ✨ Core Features

### 📊 1. Intelligent Data Profiling & EDA
* **Automated Data Quality Checks:** Instantly detects data leakage, highly correlated features, constant columns, and missing values.
* **Interactive Visualizations:** Multi-dimensional scatter plots, distribution histograms, outlier-detection box plots, and correlation heatmaps (powered by *Plotly*).
* **Dataset Health Scoring:** Generates a real-time health score reflecting the structural integrity of your dataset.

### 🛠️ 2. Smart Preprocessing & Feature Engineering
* **Auto-Imputation:** Handles missing data dynamically via Mean, Median, Mode, or Sequence Fills.
* **Smart Encoding & Scaling:** One-click Categorical Label Encoding, Standardization, and Normalization.
* **Automated Feature Engineering:** Extracts deep patterns through Polynomials, Logarithmic transformations, Binnings, and Interaction features.

### 🤖 3. The Neural Engine (Model Training)
* **Multi-Model Pipeline:** Simultaneously trains and compares 10+ algorithms, including **XGBoost, LightGBM, CatBoost, Random Forest,** and **SVM**.
* **Unsupervised Capabilities:** Implements advanced clustering techniques (K-Means, Agglomerative, Birch) visualized via PCA.
* **Hyperparameter Tuning:** Built-in `RandomizedSearchCV` for finding optimal model architectures.
* **Dynamic Leaderboard:** Auto-ranks models based on Accuracy/RMSE and automatically deploys the best-performing model.

### 🔍 4. Explainable AI (XAI) & Diagnostics
* **SHAP Integration:** Decodes complex model decisions using Global and Local SHAP (SHapley Additive exPlanations) visualizations.
* **Overfitting Detector:** Measures the performance variance between training and testing data to ensure robust generalization.
* **Advanced Metrics:** Confusion Matrices, ROC-AUC Curves, and Precision-Recall evaluation.

### 📤 5. Enterprise Deployment Features
* **Bulk Prediction Studio:** Upload an unseen CSV to generate AI predictions for thousands of rows simultaneously.
* **One-Click Export:** Download your trained Production Pipeline (`.pkl`), Best Model, Preprocessed Dataset, and Leaderboard.
* **Automated PDF Reports:** Generates a beautifully formatted, comprehensive PDF executive summary of the entire ML workflow via *ReportLab*.

---

## 💻 Technology Stack

| Category | Technologies Used |
| :--- | :--- |
| **Frontend UI** | Streamlit, Custom Premium CSS |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn, XGBoost, LightGBM, CatBoost |
| **Data Visualization** | Plotly Express, Matplotlib |
| **Model Explainability** | SHAP (Shapley Additive exPlanations) |
| **Document Generation**| ReportLab |

---

## ⚙️ Installation & Local Setup

Want to run Neuronix AI on your local machine? Follow these steps:

**1. Clone the repository:**
```bash


 👨‍💻 Developed By

**Krishna Gediya**  
*Data Scientist & AI Enthusiast*  
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=for-the-badge&logo=github)](https://github.com/krishnagediya5)
git clone [https://github.com/krishnagediya5/Neuronix-AI.git](https://github.com/krishnagediya5/Neuronix-AI.git)
cd Neuronix-AI
