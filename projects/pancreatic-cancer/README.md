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

### 4. Environment Setup & Dependency Management
To ensure fast, deterministic dependency resolution and environment reproducibility, this project uses **`uv`** alongside `pyproject.toml` and `uv.lock`.

#### Engineering Decision on Dependencies
In a typical production deployment, I usually strip out unused modeling libraries (keeping only the single winning framework) to minimize container image size and attack surface. However, for this project, **all candidate modeling and experimentation libraries (XGBoost, LightGBM, CatBoost, Scikit-Learn, and Optuna) have been intentionally retained** in the environment configuration. 

This ensures **100% end-to-end reproducibility** for reviewers—allowing anyone cloning this repository to run the training, validation, and hyperparameter tuning scripts out of the box without encountering `ModuleNotFoundError` or version mismatches.

#### Quick Start (Local Setup)
1. **Clone the repository and navigate to the project folder:**
   cd projects/pancreatic-cancer

2. **Sync the virtual environment using uv:**
   uv sync
   (if you also want to run hyperparameter optimization or test scripts, install the dev dependencies as well)

3. **Explore the notebook:**
   uv run code train.ipynb

4. **Then, run the more compact py script:**
   uv run python train.py

5. **Start the prediction service:**
   uv run uvicorn predict:app --reload --port 7860   

### 5. Preprocessing & Feature Engineering Module (`preprocess.py`)
To maintain a clean and modular codebase, all core data transformation and modeling logic is centralized in `preprocess.py`. Key components include:

- **`overview(df)`**: Generates an exploratory data summary table combining data types, unique value counts, missing value statistics, and the transposed head of the dataset.
- **`uniform(df)`**: Standardizes column names and categorical string values (converts to lowercase and replaces spaces with underscores) to prevent data mismatch bugs.
- **`add_columns(df, na_limit)`**: Automatically flags missing values by creating binary indicator columns (`_is_na`) for numeric features exceeding the missingness threshold.
- **`preprocess()`**: Builds a reusable scikit-learn `ColumnTransformer` pipeline that applies `StandardScaler` to numerical features and `OneHotEncoder` (with unknown-category handling and first-drop to avoid multicollinearity) to categorical features.
- **`objective(trial, ...)`**: A unified Optuna objective function supporting robust hyperparameter optimization for 12 different model variants (including Logistic Regression/ElasticNet, Random Forest, Extra Trees, CatBoost, LightGBM, and XGBoost) for both regression and classification tasks.

### 6. Training Reproduction (`train.py`)
While the experimentation and hyperparameter tuning across all models are documented in the Jupyter notebook, the final production-ready training pipeline has been extracted into a clean, standalone script (`train.py`). 

To train the winning model (**XGBoost** with optimized hyperparameters) from scratch and serialize the final pipeline (`model.bin`), run:

```bash
uv run python train.py
```

This script executes the core workflow:
1. **Ingestion & Cleaning:** Loads the raw dataset, standardizes columns via `preprocess.py`, and fills missing values.
2. **Feature Engineering:** Converts dates to numerical timestamps and encodes the target variable.
3. **Data Splitting & Leakage Prevention:** Applies a stratified split and removes target-leaking features (`survived`, `diagnosis_year`, `five_year_status`, `survival_months`).
4. **Preprocessing & Training:** Fits the scikit-learn transformer on the training set and trains the optimized XGBoost classifier.
5. **Artifact Generation:** Saves the complete pipeline (`preprocessor + model`) into `model.bin` for downstream inference.

### 7. Model Serving & Deployment (`predict.py` & `Dockerfile`)
To deploy the trained XGBoost model into production, the repository features a robust inference service and containerization setup:

- **Input Validation (`predict.py`):** Uses **FastAPI** and **Pydantic** (`PatientData` class) to strictly validate incoming payloads against expected medical ranges (e.g., age, BMI, CA 19-9 biomarker levels) and date constraints (2015–2026), automatically filtering out data-leaking or unnecessary fields via `extra="ignore"`.
- **API Request Testing (`request.py`):** A lightweight integration script utilizing `requests` to test and verify local or remote prediction endpoints using realistic patient payloads.
- **Containerization (`Dockerfile`):** 
  - Built on a lightweight `python:3.12.10-slim-bookworm` base image.
  - Utilizes Astral's **`uv`** binary for deterministic and fast dependency synchronization.
  - Cleans cache layers to minimize the final production image footprint.

#### Running the API Locally
```bash
uv run uvicorn predict:app --reload --port 7860
```

### Testing the Service
```bash
uv run python request.py
```

### Building and Running with Docker
```bash
# Build the container image
docker build -t pancreatic-cancer-service .

# Run the container locally
docker run -it --rm -p 9696:9696 pancreatic-cancer-service
```