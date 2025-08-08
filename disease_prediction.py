import streamlit as st
import numpy as np
import pickle

# Load all models and scalers
def load_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

kidney_model = load_pickle('random_forest_kidney_model.pkl')
kidney_scaler = load_pickle('kidney_scaler.pkl')

liver_model = load_pickle('rf_liver_model.pkl')
liver_scaler = load_pickle('scaler.pkl')

parkinsons_model = load_pickle('parkinsons_rf_model.pkl')
parkinsons_scaler = load_pickle('parkinsons_scaler.pkl')

# --- Streamlit UI ---
st.set_page_config(page_title="Multi-Disease Prediction System", layout="wide")
st.title("🧠🩺 Disease Prediction App")
st.markdown("This app predicts the presence of **Kidney**, **Liver**, or **Parkinson's** disease using trained ML models.")

# Tabs
tab1, tab2, tab3 = st.tabs(["🧪 Kidney Disease", "🧬 Liver Disease", "🧠 Parkinson’s Disease"])

# ------------------------ KIDNEY ------------------------
with tab1:
    st.header("Kidney Disease Prediction")

    kidney_inputs = {}
    kidney_features = ['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane']

    for feature in kidney_features:
        kidney_inputs[feature] = st.text_input(f"{feature}")

    if st.button("Predict Kidney Disease"):
        try:
            input_data = [float(kidney_inputs[feature]) if kidney_inputs[feature] not in ['yes', 'no'] else 1.0 if kidney_inputs[feature]=='yes' else 0.0 for feature in kidney_features]
            scaled_input = kidney_scaler.transform([input_data])
            result = kidney_model.predict(scaled_input)
            st.success("Likely Kidney Disease" if result[0]==1 else "No Kidney Disease")
        except Exception as e:
            st.error(f"Invalid input: {e}")

# ------------------------ LIVER ------------------------
with tab2:
    st.header("Liver Disease Prediction")

    liver_features = ['Age','Gender','Total_Bilirubin','Direct_Bilirubin',
                      'Alkaline_Phosphotase','Alamine_Aminotransferase',
                      'Aspartate_Aminotransferase','Total_Protiens','Albumin',
                      'Albumin_and_Globulin_Ratio']

    liver_input = []
    for feature in liver_features:
        val = st.text_input(f"{feature}", key=feature)
        liver_input.append(val)

    if st.button("Predict Liver Disease"):
        try:
            liver_input = [float(x) for x in liver_input]
            liver_input = liver_scaler.transform([liver_input])
            result = liver_model.predict(liver_input)
            st.success("Likely Liver Disease" if result[0]==1 else "No Liver Disease")
        except Exception as e:
            st.error(f"Invalid input: {e}")

# ------------------------ PARKINSONS ------------------------
with tab3:
    st.header("Parkinson’s Disease Prediction")

    parkinsons_features = ['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)',
        'MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP',
        'MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5',
        'MDVP:APQ','Shimmer:DDA','NHR','HNR','RPDE','DFA','spread1',
        'spread2','D2','PPE']

    parkinsons_input = []
    for feature in parkinsons_features:
        val = st.text_input(f"{feature}", key=feature)
        parkinsons_input.append(val)

    if st.button("Predict Parkinson’s Disease"):
        try:
            parkinsons_input = [float(x) for x in parkinsons_input]
            parkinsons_input = parkinsons_scaler.transform([parkinsons_input])
            result = parkinsons_model.predict(parkinsons_input)
            st.success("Likely Parkinson’s Disease" if result[0]==1 else "No Parkinson’s Disease")
        except Exception as e:
            st.error(f"Invalid input: {e}")