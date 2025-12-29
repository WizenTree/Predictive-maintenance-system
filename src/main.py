import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

class PredictiveMaintenance:
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.model = None
        self.scaler = StandardScaler()
        self.type_mapping = {'L': 0, 'M': 1, 'H': 2}

    def clean_data(self, df):
        """Cleans and prepares raw data."""
        # Drop identifiers
        cols_to_drop = ['UDI', 'Product ID', 'HDF', 'OSF', 'PWF', 'RNF', 'TWF']
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        # Handle Categorical 'Type'
        df['Type'] = df['Type'].astype(str).str.strip()
        df['Type'] = df['Type'].map(self.type_mapping)
        
        # Feature Engineering: Power and Temperature Difference
        # Conversion: RPM to Rad/s for physical power calculation
        df['Power'] = df['Torque [Nm]'] * (df['Rotational speed [rpm]'] * 2 * np.pi / 60)
        df['Temp_Diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
        
        return df

    def train(self, data_path):
        """Standard pipeline: Load -> Clean -> Split -> Scale -> Train"""
        df = pd.read_csv(data_path)
        df = self.clean_data(df)
        
        X = df.drop(columns=['Machine failure'])
        y = df['Machine failure']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Handle Imbalance
        ratio = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Initialize XGBoost
        self.model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            scale_pos_weight=ratio,
            random_state=42
        )
        
        print("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_probs = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_probs >= self.threshold).astype(int)
        
        print("\n--- Model Evaluation ---")
        print(classification_report(y_test, y_pred))
        
    def save_production_artifacts(self, model_name="maintenance_model.json", scaler_name="scaler.joblib"):
        """Saves artifacts in industry-standard formats."""
        # Use the internal booster to save specifically as JSON/UBJ
        self.model.get_booster().save_model(model_name)
        joblib.dump(self.scaler, scaler_name)
        print(f"Artifacts saved: {model_name}, {scaler_name}")

if __name__ == "__main__":
    # Initialize the system
    pm_system = PredictiveMaintenance(threshold=0.3)
    
    # Run training (Update path to your CSV)
    pm_system.train('data/Dataset.csv')
    
    # Save for deployment
    pm_system.save_production_artifacts()