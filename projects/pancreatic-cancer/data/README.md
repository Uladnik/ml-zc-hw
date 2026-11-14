<p align="center">
  <img src="https://img.shields.io/badge/🟣_Pancreatic_Cancer-Global_Dataset-7B2D8E?style=for-the-badge&labelColor=4A0072" alt="Pancreatic Cancer Dataset"/>
</p>

<h1 align="center">🟣 Pancreatic Cancer Global Clinical & Risk Factor Dataset</h1>
<h3 align="center">2,000 Patients · 60 Countries · 6 WHO Regions · 50+ Features · 2015–2025</h3>

<p align="center">
  <a href="https://www.kaggle.com/zkskhurram"><img src="https://img.shields.io/badge/Kaggle-zkskhurram-20BEFF?style=flat-square&logo=kaggle&logoColor=white" alt="Kaggle"/></a>
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/License-CC_BY--SA_4.0-EF9421?style=flat-square&logo=creativecommons&logoColor=white" alt="License"/>
  <img src="https://img.shields.io/badge/Records-2,000-success?style=flat-square" alt="Records"/>
  <img src="https://img.shields.io/badge/Features-50+-blue?style=flat-square" alt="Features"/>
  <img src="https://img.shields.io/badge/Updated-March_2026-red?style=flat-square" alt="Updated"/>
</p>

---

## 📋 Overview

**Pancreatic cancer** is one of the most aggressive and lethal cancers worldwide, with a **5-year survival rate of only ~12%**. It is the **3rd leading cause of cancer death** in the United States and **7th globally**. This dataset provides a comprehensive synthetic clinical dataset designed for research, EDA, machine learning, and educational purposes.

This dataset contains **2,000 patient records** spanning **2015–2025** across all **6 WHO regions** and **60 countries**, covering:

- 🧬 **Genetic Mutations** — KRAS (90%+ of PDAC), TP53, CDKN2A, SMAD4, BRCA1/2, PALB2, ATM
- 🏥 **Cancer Types** — PDAC, PNET, Acinar Cell, IPMN, SPN, Pancreatoblastoma
- 💊 **14 Treatment Modalities** — Whipple, FOLFIRINOX, Gemcitabine, Immunotherapy, Targeted Therapy
- 📍 **Anatomical Locations** — Head (60%), Body, Tail, Neck, Uncinate Process
- 🔬 **Biomarkers** — CA 19-9, CEA, CA 125 serum levels
- ⚕️ **Surgical Outcomes** — Resectability status, margin status, vascular involvement
- 📊 **50+ Clinical Features** — Demographics, risk factors, symptoms, staging, treatment, survival

---

## 📊 Dataset Files

| # | File | Description | Size |
|---|------|-------------|------|
| 1 | `pancreatic_cancer_dataset.csv` | 🟣 Main dataset — 2,000 records × 50+ columns | ~500 KB |
| 2 | `Pancreatic_Cancer_Global_Analysis.ipynb` | 📓 Comprehensive EDA notebook with 15+ visualizations | — |
| 3 | `generate_dataset.py` | 🐍 Python script to regenerate the dataset | — |
| 4 | `Pancreatic.py` | 🔬 Baseline ML model script | — |
| 5 | `dataset-metadata.json` | 📋 Kaggle dataset metadata | — |
| 6 | `README.md` | 📄 This documentation file | — |
| 7 | `requirements.txt` | 📦 Python dependencies | — |

---

## 🏗️ Key Features (50+ Columns)

### 👤 Demographics & Geography
| Column | Type | Description |
|--------|------|-------------|
| `Patient_ID` | String | Unique identifier (PC-XXXX) |
| `Diagnosis_Year` | Integer | Year of diagnosis (2015–2025) |
| `Diagnosis_Date` | Date | Full diagnosis date |
| `WHO_Region` | String | 6 WHO regions |
| `Country` | String | 60 countries worldwide |
| `Age` | Integer | Age at diagnosis (35–92) |
| `Gender` | String | Male / Female |
| `Ethnicity` | String | 7 ethnicity categories |

### 🚬 Risk Factors & Lifestyle
| Column | Type | Description |
|--------|------|-------------|
| `Smoking_Status` | String | Current / Former / Never Smoked |
| `Pack_Years` | Integer | Cumulative smoking exposure |
| `Alcohol_Use` | String | None / Light / Moderate / Heavy |
| `BMI` | Float | Body Mass Index |
| `BMI_Category` | String | Normal / Overweight / Obese |
| `Diabetes_Status` | String | Type 2 / New-Onset / No Diabetes |
| `Chronic_Pancreatitis` | String | History of chronic pancreatitis (15× risk) |
| `Family_History` | String | Family history of pancreatic cancer |
| `Hereditary_Syndrome` | String | BRCA2, Lynch, FAMMM, Peutz-Jeghers, etc. |
| `Blood_Group` | String | ABO blood group (Non-O = higher risk) |

### 🧬 Genetics & Tumor Characteristics
| Column | Type | Description |
|--------|------|-------------|
| `Genetic_Mutation` | String | KRAS, TP53, CDKN2A, SMAD4, BRCA1/2, etc. |
| `Cancer_Type` | String | PDAC, PNET, Acinar, IPMN, SPN, Pancreatoblastoma |
| `Tumor_Location` | String | Head, Body, Tail, Neck, Uncinate Process |
| `Cancer_Stage` | String | Stage I / II / III / IV |
| `Resectability` | String | Resectable, Borderline, Locally Advanced, Metastatic |
| `Tumor_Size_cm` | Float | Tumor size in centimeters |
| `Tumor_Grade` | String | Grade 1 (Well) / 2 (Moderate) / 3 (Poorly) differentiated |
| `Metastasis_Site` | String | Liver, Lungs, Peritoneum, Bones, etc. |

### 🔬 Biomarkers
| Column | Type | Description |
|--------|------|-------------|
| `Biomarker_Status` | String | CA 19-9, CEA, CA 125 elevation status |
| `CA_19_9_Level` | Float | Serum CA 19-9 level (U/mL) |

### 🤒 Symptoms (12 columns)
| Column | Description |
|--------|-------------|
| `Jaundice` | Obstructive jaundice (common in head tumors) |
| `Abdominal_Pain` | Epigastric/abdominal pain |
| `Back_Pain` | Radiating back pain |
| `Weight_Loss` | Unexplained weight loss |
| `Loss_of_Appetite` | Anorexia |
| `Nausea_Vomiting` | Nausea and vomiting |
| `New_Onset_Diabetes` | New diabetes as presenting symptom |
| `Dark_Urine` | Dark urine (biliary obstruction) |
| `Light_Colored_Stool` | Pale stools (biliary obstruction) |
| `Fatigue` | Chronic fatigue |
| `Blood_Clots` | Trousseau syndrome (VTE) |
| `Digestive_Problems` | Steatorrhea, malabsorption |

### 💊 Treatment & Surgery
| Column | Type | Description |
|--------|------|-------------|
| `Diagnosis_Method` | String | CT, MRI, EUS, ERCP, PET, Biopsy, etc. |
| `Treatment` | String | 14 treatment modalities |
| `Surgery_Performed` | String | Whether surgery was done |
| `Surgical_Margin` | String | R0 / R1 / R2 / N/A |
| `Lymph_Node_Status` | String | Positive / Negative |
| `Vascular_Involvement` | String | SMA, Celiac, Portal Vein, SMV, etc. |
| `Perineural_Invasion` | String | Characteristic of PDAC |
| `ECOG_Score` | Integer | Performance status (0–4) |

### 📈 Outcomes
| Column | Type | Description |
|--------|------|-------------|
| `Survival_Months` | Integer | Months survived post-diagnosis |
| `Survived` | String | **TARGET VARIABLE** — Yes / No |
| `Five_Year_Status` | String | Alive / Deceased / Alive (< 5 years) |

---

## 📈 Key Statistics & Findings

| Metric | Value |
|--------|-------|
| 🟣 Most Common Type | Pancreatic Ductal Adenocarcinoma (PDAC) — ~80% |
| 📍 Most Common Location | Head of Pancreas — ~60% |
| 🧬 Most Common Mutation | KRAS — ~30% (90%+ in real-world PDAC) |
| ⚠️ Stage IV at Diagnosis | ~52% of all cases |
| 💀 Overall 5-Year Survival | ~12% |
| 🔬 CA 19-9 Elevated | ~50% of Stage III/IV patients |
| 🏥 Surgery Possible | ~15-20% of cases |
| 💊 Most Common Chemo | FOLFIRINOX & Gemcitabine-based regimens |

---

## 🌍 WHO Region Coverage

| WHO Region | Countries | Key Characteristics |
|------------|-----------|---------------------|
| 🌍 Africa | Nigeria, South Africa, Kenya, Egypt, Ethiopia, Ghana, Tanzania, Morocco, Algeria, Uganda | Lower incidence, limited screening access |
| 🌎 Americas | USA, Canada, Brazil, Mexico, Argentina, Colombia, Chile, Peru, Cuba, Venezuela | High incidence, advanced treatment options |
| 🌏 South-East Asia | India, Bangladesh, Indonesia, Thailand, Myanmar, Sri Lanka, Nepal, Maldives, Timor-Leste, Bhutan | Rising incidence, diabetes co-morbidity |
| 🌍 Europe | UK, Germany, France, Italy, Spain, Russia, Turkey, Poland, Netherlands, Sweden | Highest incidence rate, strong research base |
| 🌍 Eastern Mediterranean | Pakistan, Iran, Saudi Arabia, UAE, Iraq, Jordan, Lebanon, Oman, Qatar, Kuwait | High diabetes prevalence, growing awareness |
| 🌏 Western Pacific | China, Japan, South Korea, Australia, Philippines, Vietnam, Malaysia, New Zealand, Singapore, Mongolia | Highest absolute numbers (China, Japan) |

---

## 🔬 Methodology & Provenance

### Data Generation Process

This dataset was generated using Python with **NumPy** and **Pandas**, using probability distributions carefully calibrated from published epidemiological and clinical research:

1. **Regional Incidence Weights** — Based on GLOBOCAN 2022 age-standardized incidence rates per WHO region
2. **Risk Factor Distributions** — Smoking rates, obesity prevalence, diabetes rates from WHO Global Health Observatory
3. **Cancer Type Distribution** — PDAC (~80%), PNET (~8%), others based on histopathological registries
4. **Staging Distribution** — Reflects real-world late diagnosis pattern (52% Stage IV at diagnosis)
5. **Survival Outcomes** — Calibrated to NCI SEER 5-year survival data:
   - Stage I: ~44% (localized)
   - Stage II: ~18% (regional)
   - Stage III: ~6% (locally advanced)
   - Stage IV: ~3% (metastatic)
6. **Genetic Mutations** — KRAS dominance (30%), with TP53, CDKN2A, SMAD4 as key secondary mutations
7. **Treatment Assignment** — Based on NCCN Clinical Practice Guidelines for resectability status

### Reproducibility

```bash
python generate_dataset.py  # Seed = 42 → identical output every run
```

### Limitations

- ⚠️ **Synthetic data** — Not from actual patient records
- ⚠️ Simplified distributions — Real-world data has more complexity
- ⚠️ No temporal treatment evolution — Treatment protocols don't change over the 11-year span
- ⚠️ Independence assumption — Some risk factor correlations are simplified

---

## 🎯 Use Cases

| # | Use Case | Description |
|---|----------|-------------|
| 1 | 📊 **Exploratory Data Analysis** | Risk factor distributions, demographic patterns, geographic analysis |
| 2 | 🤖 **Machine Learning — Classification** | Predict survival outcome (Survived: Yes/No) |
| 3 | 📈 **Survival Analysis** | Kaplan-Meier curves, Cox proportional hazards regression |
| 4 | 🧬 **Genetic Analysis** | Mutation frequency by cancer type, stage, and demographics |
| 5 | 🌍 **Geographic Analysis** | Regional incidence, treatment access, outcome disparities |
| 6 | 💊 **Treatment Effectiveness** | Compare outcomes across treatment modalities by stage |
| 7 | 🔬 **Biomarker Analysis** | CA 19-9 correlation with stage, prognosis, treatment response |
| 8 | 🏥 **Clinical Decision Support** | Risk stratification, screening strategy optimization |
| 9 | 📚 **Academic Research** | Teaching clinical data analysis, epidemiology, biostatistics |
| 10 | 📱 **Dashboard Development** | Interactive dashboards with Plotly, Streamlit, Power BI |

---

## 🚀 Quick Start

### Option 1: Using .kag_Go Virtual Environment (Recommended)

```bash
# Navigate to project directory
cd Kaggle/DataSets/Health/Pancreatic_Cancer

# Create and activate virtual environment
python -m venv .kag_Go
.kag_Go\Scripts\activate          # Windows
# source .kag_Go/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Generate dataset
python generate_dataset.py

# Launch notebook
jupyter notebook Pancreatic_Cancer_Global_Analysis.ipynb
```

### Option 2: Quick Load in Python

```python
import pandas as pd

# Load dataset
df = pd.read_csv('pancreatic_cancer_dataset.csv', parse_dates=['Diagnosis_Date'])

# Quick overview
print(f"📊 Shape: {df.shape}")
print(f"🌍 WHO Regions: {df.WHO_Region.nunique()}")
print(f"🏥 Cancer Types: {df.Cancer_Type.nunique()}")
print(f"⚠️ Stage IV: {(df.Cancer_Stage == 'Stage IV').mean():.1%}")
print(f"💀 Survival Rate: {(df.Survived == 'Yes').mean():.1%}")
```

### Option 3: Kaggle Notebook

```python
# On Kaggle
df = pd.read_csv('/kaggle/input/pancreatic-cancer-global-clinical-dataset/pancreatic_cancer_dataset.csv')
```

---

## 📚 Data Sources & Citations

| # | Source | Organization | Usage |
|---|--------|-------------|-------|
| 1 | [GLOBOCAN 2022](https://gco.iarc.fr/) | IARC / WHO | Regional incidence rates, global cancer statistics |
| 2 | [ACS Pancreatic Cancer](https://www.cancer.org/cancer/types/pancreatic-cancer.html) | American Cancer Society | Risk factors, staging, survival rates |
| 3 | [PanCAN](https://www.pancan.org/) | Pancreatic Cancer Action Network | Symptoms, genetic mutations, biomarkers |
| 4 | [NCI SEER](https://seer.cancer.gov/) | National Cancer Institute | 5-year survival data, historical trends |
| 5 | [WHO GHO](https://www.who.int/data/gho) | World Health Organization | Country-level cancer mortality, risk factors |
| 6 | [The Lancet GH](https://www.thelancet.com/journals/langas/home) | The Lancet | Peer-reviewed clinical research |
| 7 | [NCCN Guidelines](https://www.nccn.org/) | National Comprehensive Cancer Network | Treatment protocols, resectability criteria |

---

## 👤 Author & Mentor

<table>
<tr>
<td align="center">

### 👨‍💻 Author
**Khurram Shahzad**<br/>
Data Scientist & AI Enthusiast<br/>
[![Kaggle](https://img.shields.io/badge/Kaggle-zkskhurram-20BEFF?style=flat-square&logo=kaggle)](https://www.kaggle.com/zkskhurram)

</td>
<td align="center">

### 🎓 Mentor
**Dr. Aammar Tufail**<br/>
AI & Data Science Mentor<br/>
[![YouTube](https://img.shields.io/badge/YouTube-Dr_Aammar_Tufail-FF0000?style=flat-square&logo=youtube)](https://www.youtube.com/@AammarTufail)

</td>
</tr>
</table>

---

## ⚖️ License

This dataset is released under the **[Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/)** license.

You are free to:
- ✅ **Share** — Copy and redistribute the material
- ✅ **Adapt** — Remix, transform, and build upon the material
- ⚠️ **Attribution** — Give appropriate credit, provide a link to the license
- ⚠️ **ShareAlike** — Distribute contributions under the same license

---

## 📌 Tags

`pancreatic-cancer` · `PDAC` · `PNET` · `oncology` · `healthcare` · `medical-dataset` · `cancer-prediction` · `KRAS` · `TP53` · `BRCA2` · `CA-19-9` · `whipple-procedure` · `FOLFIRINOX` · `gemcitabine` · `survival-analysis` · `WHO` · `GLOBOCAN` · `EDA` · `machine-learning` · `classification` · `clinical-data` · `public-health` · `gastrointestinal-cancer` · `biomarkers` · `risk-factors` · `diabetes` · `chronic-pancreatitis`

---

<p align="center">
  <b>🟣 If you find this dataset useful, please upvote it on Kaggle! ⬆️</b><br/>
  <i>Made with 💜 for the data science & medical research community</i>
</p>
