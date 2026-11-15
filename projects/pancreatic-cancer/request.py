import requests

# Define the prediction service URL
url = 'http://localhost:7860/predict'

# Define the sample patient data payload matching the Pydantic schema
patient = {
    "Patient_ID": "PC-0042",
    "Diagnosis_Year": 2023,
    "Diagnosis_Date": "2023-04-08",
    "WHO_Region": "Europe",
    "Country": "Germany",
    "Age": 58,
    "Gender": "Male",
    "Ethnicity": "White/Caucasian",
    "Smoking_Status": "Current Smoker",
    "Pack_Years": 25,
    "Alcohol_Use": "Moderate",
    "BMI": 26.5,
    "BMI_Category": "Overweight",
    "Diabetes_Status": "No Diabetes",
    "Chronic_Pancreatitis": "No",
    "Family_History": "Yes",
    "Hereditary_Syndrome": "Lynch Syndrome",
    "Blood_Group": "A",
    "Genetic_Mutation": "KRAS",
    "Cancer_Type": "Pancreatic Ductal Adenocarcinoma (PDAC)",
    "Tumor_Location": "Head",
    "Cancer_Stage": "Stage II",
    "Resectability": "Resectable",
    "Tumor_Size_cm": 3.2,
    "Tumor_Grade": "Grade 2 (Moderately Differentiated)",
    "Metastasis_Site": "No Distant Metastasis",
    "Biomarker_Status": "CA 19-9 Elevated",
    "CA_19_9_Level": 145.5,
    "Jaundice": "Yes",
    "Abdominal_Pain": "Yes",
    "Back_Pain": "No",
    "Weight_Loss": "Yes",
    "Loss_of_Appetite": "No",
    "Nausea_Vomiting": "No",
    "New_Onset_Diabetes": "No",
    "Dark_Urine": "Yes",
    "Light_Colored_Stool": "No",
    "Fatigue": "Yes",
    "Blood_Clots": "No",
    "Digestive_Problems": "Yes",
    "Diagnosis_Method": "CT Scan",
    "Treatment": "Whipple Procedure (Pancreaticoduodenectomy)",
    "Surgery_Performed": "Yes",
    "Surgical_Margin": "R0 (Negative)",
    "Lymph_Node_Status": "Negative",
    "Vascular_Involvement": "Portal Vein",
    "Perineural_Invasion": "No",
    "ECOG_Score": 1
}

# Send a POST request with JSON payload to the FastAPI prediction service
response = requests.post(url, json=patient)

# Output the HTTP status code and raw response text
print("Status code:", response.status_code)
print("Response body:", response.text)

# Parse and print JSON response if the body is not empty
if response.text: 
    print(response.json())