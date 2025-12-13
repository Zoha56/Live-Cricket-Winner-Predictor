import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_and_save_model():
    # Load data
    try:
        df = pd.read_csv('ball_by_ball_it20.csv')
        df = df.drop(columns=['Unnamed: 0'])
    except FileNotFoundError:
        print("Error: Dataset 'ball_by_ball_it20.csv' not found.")
        return

    # Filter for 2nd innings data and drop NaNs in target columns
    df_2nd_innings = df[df['Innings'] == 2].copy()
    df_2nd_innings.dropna(subset=['Runs to Get'], inplace=True)
    
    # Feature Engineering/Cleaning
    df_2nd_innings['overs_faced'] = 120 - df_2nd_innings['Balls Remaining']
    
    # Current Run Rate (CRR)
    df_2nd_innings['current_run_rate'] = np.where(
        df_2nd_innings['overs_faced'] > 0, 
        (df_2nd_innings['Innings Runs'] / df_2nd_innings['overs_faced']) * 6, 
        0
    )
    
    # Required Run Rate (RRR)
    df_2nd_innings['req_run_rate'] = np.where(
        df_2nd_innings['Balls Remaining'] > 0, 
        (df_2nd_innings['Runs to Get'] / df_2nd_innings['Balls Remaining']) * 6, 
        0
    )
    
    # Select Features
    categorical_features = ['Venue', 'Bat First', 'Bat Second']
    numerical_features = ['Runs to Get', 'Balls Remaining', 'Innings Wickets', 'current_run_rate', 'req_run_rate']
    
    X = df_2nd_innings[categorical_features + numerical_features]
    y = df_2nd_innings['Chased Successfully']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # One Hot Encoding Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Train Logistic Regression Model
    model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
    model.fit(X_train_processed, y_train)
    
    # Evaluate 
    X_test_processed = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1] # Probability for the positive class (1)
    
    # --- CALCULATE ALL METRICS ---
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    }
    # -----------------------------
    
    print(f"Model Accuracy: {metrics['Accuracy']:.4f}")
    
    # Save the Model, Preprocessor, and Metrics using pickle (.pkl)
    
    # 1. Save the trained model
    with open('cricket_predictor_model.pkl', 'wb') as file:
        pickle.dump(model, file)
        
    # 2. Save the preprocessor (Contains OHE encoder and categories)
    with open('model_preprocessor.pkl', 'wb') as file:
        pickle.dump(preprocessor, file)
        
    # 3. Save the performance metrics
    with open('model_metrics.pkl', 'wb') as file:
        pickle.dump(metrics, file)
        
    print("\nTraining complete.")
    print("Model saved as cricket_predictor_model.pkl")
    print("Preprocessor (with categorical mappings) saved as model_preprocessor.pkl")
    print("Metrics saved as model_metrics.pkl")

if __name__ == '__main__':
    train_and_save_model()