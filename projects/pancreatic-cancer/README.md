# Pancreatic Cancer Risk Prediction
## Midterm Project — Machine Learning Zoomcamp

### 1. Problem Description & Motivation
Pancreatic cancer is widely recognized as one of the most aggressive and lethal malignancies, frequently diagnosed at advanced stages where therapeutic options are severely limited. Early identification of high-risk individuals and timely clinical intervention are critical for improving patient outcomes and survival rates.

This project holds deep personal significance for my family, as we have been personally touched by this disease. Driven by this experience, I wanted to channel a personal challenge into a data-driven mission—building a predictive machine learning model to better understand and assess pancreatic cancer risk factors based on clinical data.

### 2. Dataset
For this project, I am using the **Pancreatic Cancer Global Clinical and Risk Factor** dataset sourced from Kaggle. It contains a rich mix of clinical attributes, patient demographics, and specific risk factors, making it an ideal foundation for training a robust classification model.

### 3. Project Roadmap
Following a structured machine learning engineering pipeline, this project covers:
- **Exploratory Data Analysis (EDA):** Inspecting distributions, cleaning data, and analyzing feature importance and correlations.
- **Model Training & Optimization:** Building and comparing baseline models with advanced tree-based algorithms (XGBoost/CatBoost/LightGBM), fine-tuned using Optuna.
- **Model Deployment:** Wrapping the trained model into a production-ready web service using FastAPI and Pydantic, and containerizing it with Docker for seamless reproducibility.