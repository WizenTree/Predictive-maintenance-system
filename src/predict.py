import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

model = XGBClassifier()
model.load_model("maintenance_model.json")
scaler = joblib.load("scaler.joblib")

def real_time_prediction(raw_data, threshold=0.3):
    """
    Simulates a machine sensor reading coming into the system.
    raw_data: dictionary of sensor values
    """
    df = pd.DataFrame([raw_data])

    df['Power'] = df['Torque [Nm]'] * (df['Rotational speed [rpm]'] * 2 * np.pi / 60)
    df['Temp_Diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    
    features = ['Type', 'Air temperature [K]', 'Process temperature [K]', 
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 
                'Power', 'Temp_Diff']
    
    X_input = df[features]
    
    X_scaled = scaler.transform(X_input)
    
    prob = model.predict_proba(X_scaled)[:, 1][0]
    prediction = 1 if prob >= threshold else 0
    
    return prediction, prob

new_machine_data = {
    'Type': 0, # Low Quality
    'Air temperature [K]': 300.1,
    'Process temperature [K]': 310.2,
    'Rotational speed [rpm]': 1400,
    'Torque [Nm]': 55.5,  # High Torque
    'Tool wear [min]': 210 # High Wear
}

status, risk_score = real_time_prediction(new_machine_data)
print(f"Prediction: {'FAILURE RISK' if status == 1 else 'HEALTHY'}")
print(f"Risk Probability: {risk_score:.2f}")