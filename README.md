
# 🏡 House Price Prediction System — ZenML + MLflow MLOps Pipeline  

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-orange)](https://mlflow.org/)
[![ZenML](https://img.shields.io/badge/ZenML-MLOps_Pipeline-green)](https://zenml.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML_Modeling-yellow)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

---

## 🚀 Overview  

This project demonstrates an **end-to-end MLOps workflow** for predicting house prices using **ZenML** for pipeline orchestration and **MLflow** for experiment tracking.  
It takes a standard machine learning problem — *House Price Prediction* — and elevates it into a **production-ready ML system**.  

---

## 🧩 Tech Stack  

| Tool / Framework | Purpose |
|------------------|----------|
| 🐍 **Python** | Core programming language |
| 📊 **Pandas, NumPy** | Data preprocessing and transformation |
| 🤖 **Scikit-learn** | Model training and evaluation |
| 🧠 **MLflow** | Experiment tracking and model registry |
| ⚙️ **ZenML** | Pipeline orchestration and reproducibility |
| 💻 **Streamlit (optional)** | UI for serving predictions |

---

## 🏗️ Project Architecture  

```bash
House_Price_Prediction/
├── data/
│   ├── raw/                 # Raw housing data
│   └── processed/           # Cleaned and transformed data
├── src/
│   ├── data_loader.py       # Loads dataset
│   ├── preprocess.py        # Data cleaning and feature scaling
│   ├── train_model.py       # Model training logic
│   ├── evaluate_model.py    # Model evaluation
│   └── predict.py           # Inference using saved model
├── pipelines/
│   └── house_pipeline.py    # ZenML pipeline definition
├── mlruns/                  # MLflow experiment logs
├── requirements.txt         # Dependencies
└── README.md
````

---

## 🔁 ZenML Pipeline Workflow

**ZenML Pipeline** ensures every step of your ML lifecycle is reproducible and trackable.

1. 🧾 **Data Loader Step** → Loads and splits the dataset
2. 🧹 **Preprocessing Step** → Cleans data and prepares features
3. ⚙️ **Trainer Step** → Trains multiple models (Linear Regression, Random Forest, etc.)
4. 📈 **Evaluator Step** → Logs metrics (MAE, RMSE, R²) into MLflow
5. 🚀 **Deployment Step** → Pushes the best model for serving

Run the pipeline:

```bash
zenml pipeline run house_price_pipeline
```

---

## 📊 Experiment Tracking with MLflow

Launch MLflow UI to visualize experiments:

```bash
mlflow ui
```

Then open: 👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

You can:

* Compare model performance
* Track hyperparameters and metrics
* Access saved model artifacts

📸 *Example Dashboard:*

```
Experiment: house_price_prediction
 ├── Run 1: Linear Regression → RMSE: 2.85
 ├── Run 2: Random Forest → RMSE: 1.78 ✅
 └── Run 3: XGBoost → RMSE: 1.95
```

---

## 🌍 Model Serving

Once your model is logged, serve it locally:

```bash
mlflow models serve -m "runs:/<your_run_id>/model" --port 5001
```

Then send a prediction request:

```bash
curl -X POST -H "Content-Type: application/json" \
  --data '{"columns":["feature1","feature2","feature3"],"data":[[value1,value2,value3]]}' \
  http://127.0.0.1:5001/invocations
```

---

## ⚙️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/Anshul-Raj-S-V/House_prediction_system-Zenml-MLFLOW.git
cd House_prediction_system-Zenml-MLFLOW

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # (on Windows)

# Install dependencies
pip install -r requirements.txt

# Initialize ZenML repository
zenml init

# Run pipeline
python test.py
```

---

## 💡 Key Learnings

✨ Reproducible ML workflows with **ZenML pipelines**
✨ Model tracking, versioning, and registry using **MLflow**
✨ End-to-end **MLOps implementation** from data to deployment
✨ Hands-on with **model serving and API inference**

---

## 🔮 Future Enhancements

* 🔧 Integrate **Docker** for containerized deployment
* 🚀 Deploy model via **AWS Sagemaker** or **Google Cloud Vertex AI**
* ⚡ Add **Hyperparameter Optimization** (Optuna integration)
* 🧠 Include **feature importance visualization** and **Streamlit dashboard**

```

---

Would you like me to **add a small ASCII architecture diagram** (like “Data → Train → Evaluate → Deploy”) and badges like “Built with ❤️ using ZenML & MLflow”?  
It’ll make your GitHub page *look like a professional open-source project*.
```
