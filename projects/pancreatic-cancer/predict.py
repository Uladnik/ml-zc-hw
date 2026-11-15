import pickle
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI
from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

# ==========================================
# 1. Pydantic Schema for Input Data Validation
# ==========================================
class PatientData(BaseModel):
    # Model configuration: ignore extra fields (like ID) and allow population by alias
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    diagnosis_date: Optional[str] = Field(
        None, 
        pattern=r"^(201[5-9]|202[0-6])-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$",
        alias="Diagnosis_Date"
    )

    who_region: Optional[Literal[
        'Western Pacific', 'Americas', 'Europe', 
        'South-East Asia', 'Eastern Mediterranean', 'Africa'
    ]] = Field(None, alias="WHO_Region")
    
    country: Optional[Literal[
        'China', 'Mexico', 'Germany', 'Timor-Leste', 'Turkey', 'Japan', 
        'Singapore', 'Netherlands', 'Maldives', 'USA', 'Indonesia', 'Jordan', 
        'Chile', 'Vietnam', 'Algeria', 'Mongolia', 'Egypt', 'Spain', 
        'Australia', 'Philippines', 'Sri Lanka', 'Russia', 'South Africa', 
        'South Korea', 'Nepal', 'Cuba', 'UK', 'Italy', 'Lebanon', 'Qatar', 
        'Sweden', 'Brazil', 'Canada', 'Malaysia', 'Iran', 'Colombia', 
        'Argentina', 'New Zealand', 'Kuwait', 'UAE', 'India', 'Nigeria', 
        'Pakistan', 'Poland', 'France', 'Peru', 'Venezuela', 'Bhutan', 
        'Bangladesh', 'Ghana', 'Ethiopia', 'Iraq', 'Kenya', 'Myanmar', 
        'Uganda', 'Thailand', 'Oman', 'Saudi Arabia', 'Tanzania', 'Morocco'
    ]] = Field(None, alias="Country")

    age: Optional[int] = Field(None, ge=0, le=120, alias="Age")
    gender: Optional[Literal['Female', 'Male', 'Other']] = Field(None, alias="Gender")
    
    ethnicity: Optional[Literal[
        'White/Caucasian', 'South Asian', 'Black/African', 
        'Other', 'Asian', 'Middle Eastern', 'Hispanic/Latino'
    ]] = Field(None, alias="Ethnicity")
    
    smoking_status: Optional[Literal['Current Smoker', 'Former Smoker', 'Never Smoked']] = Field(None, alias="Smoking_Status")
    pack_years: Optional[int] = Field(None, ge=0, le=120, alias="Pack_Years")
    alcohol_use: Optional[Literal['Heavy', 'Moderate', 'Light']] = Field(None, alias="Alcohol_Use")
    bmi: Optional[float] = Field(None, ge=12.0, le=60, alias="BMI")
    bmi_category: Optional[Literal['Overweight', 'Normal', 'Obese']] = Field(None, alias="BMI_Category")
    diabetes_status: Optional[Literal['No Diabetes', 'New-Onset Diabetes', 'Type 2 Diabetes']] = Field(None, alias="Diabetes_Status")
    
    chronic_pancreatitis: Optional[Literal['No', 'Yes']] = Field(None, alias="Chronic_Pancreatitis")
    family_history: Optional[Literal['No', 'Yes']] = Field(None, alias="Family_History")
    
    hereditary_syndrome: Optional[Literal[
        'Lynch Syndrome', 'Li-Fraumeni Syndrome', 
        'Hereditary Pancreatitis', 'BRCA2 Mutation', 
        'Familial Atypical Multiple Mole Melanoma (FAMMM)', 'Peutz-Jeghers Syndrome'
    ]] = Field(None, alias="Hereditary_Syndrome")
    
    blood_group: Optional[Literal['B', 'A', 'O', 'AB']] = Field(None, alias="Blood_Group")
    
    genetic_mutation: Optional[Literal[
        'KRAS', 'CDKN2A', 'No Mutation Detected', 'MLH1', 'TP53', 
        'SMAD4', 'BRCA1', 'BRCA2', 'PALB2', 'ATM', 'STK11'
    ]] = Field(None, alias="Genetic_Mutation")
    
    cancer_type: Optional[Literal[
        'Pancreatic Ductal Adenocarcinoma (PDAC)', 
        'Pancreatic Neuroendocrine Tumor (PNET)', 
        'Solid Pseudopapillary Neoplasm', 'Pancreatoblastoma', 
        'Intraductal Papillary Mucinous Neoplasm (IPMN)', 'Acinar Cell Carcinoma'
    ]] = Field(None, alias="Cancer_Type")
    
    tumor_location: Optional[Literal['Head', 'Tail', 'Body', 'Neck', 'Uncinate Process']] = Field(None, alias="Tumor_Location")
    cancer_stage: Optional[Literal['Stage III', 'Stage II', 'Stage IV', 'Stage I']] = Field(None, alias="Cancer_Stage")
    resectability: Optional[Literal['Locally Advanced', 'Metastatic', 'Resectable', 'Borderline Resectable']] = Field(None, alias="Resectability")
    tumor_size_cm: Optional[float] = Field(None, ge=0.5, le=12.3, alias="Tumor_Size_cm")
    
    tumor_grade: Optional[Literal[
        'Grade 2 (Moderately Differentiated)', 
        'Grade 3 (Poorly Differentiated)', 'Grade 1 (Well Differentiated)'
    ]] = Field(None, alias="Tumor_Grade")
    
    metastasis_site: Optional[Literal[
        'Liver', 'No Distant Metastasis', 'Bones', 'Peritoneum', 
        'Multiple Sites', 'Lungs', 'Liver + Peritoneum', 'Liver + Lungs', 'Distant Lymph Nodes'
    ]] = Field(None, alias="Metastasis_Site")
    
    biomarker_status: Optional[Literal[
        'CA 19-9 Elevated', 'CEA Elevated', 'CA 125 Elevated', 
        'CA 19-9 + CEA Elevated', 'Normal Biomarkers'
    ]] = Field(None, alias="Biomarker_Status")
    
    ca_19_9_level: Optional[float] = Field(None, ge=2.0, le=25402.2, alias="CA_19_9_Level")
    
    jaundice: Optional[Literal['Yes', 'No']] = Field(None, alias="Jaundice")
    abdominal_pain: Optional[Literal['Yes', 'No']] = Field(None, alias="Abdominal_Pain")
    back_pain: Optional[Literal['Yes', 'No']] = Field(None, alias="Back_Pain")
    weight_loss: Optional[Literal['Yes', 'No']] = Field(None, alias="Weight_Loss")
    loss_of_appetite: Optional[Literal['No', 'Yes']] = Field(None, alias="Loss_of_Appetite")
    nausea_vomiting: Optional[Literal['No', 'Yes']] = Field(None, alias="Nausea_Vomiting")
    new_onset_diabetes: Optional[Literal['No', 'Yes']] = Field(None, alias="New_Onset_Diabetes")
    dark_urine: Optional[Literal['Yes', 'No']] = Field(None, alias="Dark_Urine")
    light_colored_stool: Optional[Literal['Yes', 'No']] = Field(None, alias="Light_Colored_Stool")
    fatigue: Optional[Literal['Yes', 'No']] = Field(None, alias="Fatigue")
    blood_clots: Optional[Literal['No', 'Yes']] = Field(None, alias="Blood_Clots")
    digestive_problems: Optional[Literal['Yes', 'No']] = Field(None, alias="Digestive_Problems")
    
    diagnosis_method: Optional[Literal[
        'Endoscopic Ultrasound (EUS)', 'Laparoscopy', 'MRI', 
        'Abdominal Ultrasound', 'CT Scan', 'Biopsy', 'Blood Test (CA 19-9)', 'ERCP', 'PET Scan'
    ]] = Field(None, alias="Diagnosis_Method")
    
    treatment: Optional[Literal[
        'Chemotherapy (FOLFIRINOX)', 'Chemotherapy (Gemcitabine)', 
        'Targeted Therapy (Erlotinib)', 'Distal Pancreatectomy', 'Immunotherapy', 
        'Clinical Trial', 'Palliative Care', 'Chemoradiation', 'Neoadjuvant Chemotherapy + Surgery', 
        'Chemotherapy (Gemcitabine + nab-Paclitaxel)', 'Targeted Therapy (Olaparib)', 
        'Whipple Procedure (Pancreaticoduodenectomy)', 'Radiation Therapy', 'Total Pancreatectomy'
    ]] = Field(None, alias="Treatment")
    
    surgery_performed: Optional[Literal['No', 'Yes']] = Field(None, alias="Surgery_Performed")
    surgical_margin: Optional[Literal['R1 (Microscopic Positive)', 'R2 (Macroscopic Positive)', 'R0 (Negative)']] = Field(None, alias="Surgical_Margin")
    lymph_node_status: Optional[Literal['Negative', 'Positive']] = Field(None, alias="Lymph_Node_Status")
    
    vascular_involvement: Optional[Literal[
        'Multiple Vessels', 'SMV (Superior Mesenteric Vein)', 
        'Portal Vein', 'Celiac Axis', 'SMA (Superior Mesenteric Artery)'
    ]] = Field(None, alias="Vascular_Involvement")
    
    perineural_invasion: Optional[Literal['Yes', 'No']] = Field(None, alias="Perineural_Invasion")
    ecog_score: Optional[int] = Field(None, ge=0, le=4, alias="ECOG_Score")    


class PredictResponse(BaseModel):
    """API response schema containing prediction probability."""
    prediction: float


# ==========================================
# 2. App Initialization & Model Loading
# ==========================================
app = FastAPI(title="pancreatic-cancer-survival")

# Load the serialized pipeline (preprocessor + model)
with open('model.bin', "rb") as f_in:
    preprocessor, model = pickle.load(f_in)


def prepare_data(patient: PatientData) -> pd.DataFrame:
    """
    Transforms validated patient data into a DataFrame, 
    converts keys and string values to lowercase/snake_case matching training, 
    and converts the date to a numeric timestamp.
    """
    data = {
        k.lower().replace(" ", "_"): (
            v.lower().replace(" ", "_") if isinstance(v, str) else v
        )
        for k, v in patient.model_dump().items()
    }
    df = pd.DataFrame([data])
    
    # Convert date to timestamp (consistent with training pipeline)
    if "diagnosis_date" in df.columns and df["diagnosis_date"].iloc[0] is not None:
        df["diagnosis_date"] = pd.to_datetime(df["diagnosis_date"]).astype("int64") // 10**9
        
    return df


# ==========================================
# 3. Inference Endpoint
# ==========================================
@app.post("/predict")
def predict(patient: PatientData) -> PredictResponse:
    """Accepts patient payload, runs preprocessing, and returns prediction probability."""
    df = prepare_data(patient)
    X = preprocessor.transform(df)
    response = float(model.predict_proba(X)[0][1])
    return PredictResponse(prediction=response)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)