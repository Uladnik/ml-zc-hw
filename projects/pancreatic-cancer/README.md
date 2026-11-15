# Pancreatic Cancer Survival Prediction (version 1.0)
## Midterm Project — Machine Learning Zoomcamp

### 1. Problem Description & Motivation
Pancreatic cancer is widely recognized as one of the most aggressive and lethal malignancies, frequently diagnosed at advanced stages where therapeutic options are severely limited. Early identification of high-risk individuals and timely clinical intervention are critical for improving patient outcomes and survival rates.

This project holds deep personal significance for my family, as we have been personally touched by this disease. Driven by this experience, I wanted to channel a personal challenge into a data-driven mission—building a predictive machine learning model to better understand and assess pancreatic cancer risk factors based on clinical data. The model does not provide recommendations, but it clearly shows how to increase the patient’s chances of survival by changing individual factors.

I also plan to create version 2.0 with a web interface, stable hosting, and an improved predictive model. It will be located in a separate repository. I want to create a free and effective open‑source service available to people around the world. It's not about making money, but rather about making a positive impact on people's lives.

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
```bash
cd projects/pancreatic-cancer
```

2. **Sync the virtual environment using uv:**
```bash
uv sync
```
(if you also want to run hyperparameter optimization or test scripts, install the dev dependencies as well)

3. **Explore the notebook:**
```bash
uv run code train.ipynb
```

4. **Then, run the more compact py script:**
```bash
uv run python train.py
```

5. **Start the prediction service:**
```bash
uv run uvicorn predict:app --reload --port 7860   
```

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
docker build -t pancreatic-cancer-survival-v1.0 .

# Run the container locally
docker run -it --rm -p 7860:7860 pancreatic-cancer-survival-v1.0
```

### 8. Cloud Deployment Instructions

This project is fully containerized and ready for cloud deployment. Due to recent platform policy changes regarding free Docker tiers on Hugging Face Spaces and credit card requirements on Render, here are the official configuration guides for deploying this service.

#### Option A: Deployment on Render (via Docker)
Render supports Docker-based deployments, though an initial account verification may be required.

1. **Log in to Render**: Go to [render.com](https://render.com/) and create a Web Service connected to your GitHub repository.
2. **Configure Service Settings**:
   * **Name**: `pancreatic-cancer-api` (or your preferred name)
   * **Region**: Choose your closest region (e.g., Frankfurt)
   * **Root Directory**: `projects/pancreatic-cancer`
   * **Runtime**: `Docker`
   * **Instance Type**: `Free`
3. **Advanced Docker Settings**:
   * **Build Context Directory**: `projects/pancreatic-cancer`
   * **Dockerfile Path**: `projects/pancreatic-cancer/Dockerfile`
4. **Deploy**: Click **Create Web Service**. Render will automatically build the image using the `Dockerfile` and provide a public URL once live.

---

#### Option B: Deployment on Hugging Face Spaces (via Docker)
Hugging Face provides reliable hosting for ML applications.

1. **Create a Space**: Go to [huggingface.co](https://huggingface.co/), click on your profile, and select **New Space**.
2. **Space Configuration**:
   * **Space name**: `pancreatic-cancer-api`
   * **License**: `mit`
   * **Space SDK**: Select **Docker** (Note: Ensure your account tier supports Docker spaces if using free-tier restrictions, or use a Static/Gradio template fallback).
   * **Visibility**: `Public`
3. **Upload Project Files**: 
   Upload the contents of `projects/pancreatic-cancer/` directly to the Space repository (ensuring `Dockerfile`, `predict.py`, `model.bin`, and dependency files are in the root of the Space).
4. **Port Configuration**:
   Make sure your `Dockerfile` exposes and listens on port `7860` (Hugging Face default):
```dockerfile
EXPOSE 7860
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "7860"]
```

#### Testing the Deployed Service (`request.py`)

Once your service is deployed on Render or Hugging Face, you will receive a permanent public URL (e.g., `https://your-app-name.onrender.com` or `https://huggingface.co/spaces/...`).

To test the remote API, update the `url` variable inside your local `request.py` (or `test_predict.py`) script to point to your cloud endpoint with the `/predict` suffix:

```python
# request.py 
import requests

# Replace with your actual cloud service URL
url = 'https://your-app-name.onrender.com/predict'

patient = {
    "age": 60,
    "gender": "female",
    "bmi": 25.5,
    # ... other required features ...
}

response = requests.post(url, json=patient).json()
print(response)
```

Run the script locally to verify communication with the cloud:
```bash
python request.py
```

### 9. Conclusion

This midterm project successfully demonstrates an end-to-end machine learning engineering workflow—from exploratory data analysis and rigorous hyperparameter tuning (via Optuna) to production-grade model serving (FastAPI/Pydantic), containerization (Docker with `uv`), and cloud deployment architecture. 

In **Version 2.0**, I plan to build upon this foundation by developing a full-stack web interface and establishing stable, independent hosting to create a free and effective open-source service accessible to people worldwide.