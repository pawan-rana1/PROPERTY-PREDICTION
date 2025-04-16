

# app.py
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load and preprocess data
def load_data():
    df = pd.read_csv('https://drive.google.com/file/d/1-UuhQCtJZTAUlv37fUnM-r78XdQlun5x/view?usp=drive_link')
    
    # Handle missing values
    df['Heat treatment'] = df['Heat treatment'].fillna('unknown')
    df = df.drop(['Std', 'ID', 'pH', 'Desc', 'HV'], axis=1)
    
    # Encode categorical variables
    le = LabelEncoder()
    df['Heat treatment'] = le.fit_transform(df['Heat treatment'])
    df['Material'] = le.fit_transform(df['Material'])
    
    return df, le

# Train models
def train_models(df):
    X = df.drop(['Ro', 'mu', 'G', 'Heat treatment'], axis=1)
    y_reg = df[['Ro', 'mu', 'G']]
    y_clf = df['Heat treatment']
    
    # Split data
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42)
    
    # Regression model
    reg_model = RandomForestRegressor(n_estimators=100)
    reg_model.fit(X_train, y_reg_train)
    
    # Classification model
    clf_model = RandomForestClassifier(n_estimators=100)
    clf_model.fit(X_train, y_clf_train)
    
    return reg_model, clf_model

# Streamlit app
def main():
    st.title("Material Properties Predictor")
    
    df, le = load_data()
    reg_model, clf_model = train_models(df)
    
    # User input
    st.sidebar.header("Input Material Properties")
    inputs = {
        'Material': st.sidebar.selectbox('Material', df['Material'].unique()),
        'Su': st.sidebar.number_input('Tensile Strength (Su)'),
        'Sy': st.sidebar.number_input('Yield Strength (Sy)'),
        'A5': st.sidebar.number_input('Elongation (A5)'),
        'Bhn': st.sidebar.number_input('Brinell Hardness (Bhn)'),
        'E': st.sidebar.number_input("Young's Modulus (E)")
    }
    
    # Prediction
    if st.sidebar.button('Predict'):
        input_df = pd.DataFrame([inputs])
        
        # Predict heat treatment
        heat_treatment = clf_model.predict(input_df)[0]
        heat_treatment = le.inverse_transform([heat_treatment])[0]
        
        # Predict other properties
        reg_pred = reg_model.predict(input_df)[0]
        
        # Display results
        st.subheader("Predictions")
        st.write(f"Heat Treatment: {heat_treatment}")
        st.write(f"Density (Ro): {reg_pred[0]:.2f} kg/m³")
        st.write(f"Poisson's Ratio (μ): {reg_pred[1]:.3f}")
        st.write(f"Shear Modulus (G): {reg_pred[2]:.2f} GPa")

if __name__ == '__main__':
    main()
