# 🧠 Multi-Disease Prediction Web App (Kidney, Liver & Parkinson's)

This Streamlit-based web application allows users to predict the likelihood of **Kidney Disease**, **Liver Disease**, and **Parkinson’s Disease** based on clinical and diagnostic inputs. Each disease model is hosted in a separate tab for user convenience.

## 🚀 Demo

Live App: [Click to Visit](https://multiplediseaseprediction-abcd123.streamlit.app/)  


---

## 📊 Features

- 🔬 **Kidney Disease Prediction**
- 🧬 **Liver Disease Prediction**
- 🧠 **Parkinson's Disease Prediction**
- 📈 Multi-tab layout
- ✅ Interactive UI for patient data entry
- 🔒 Backend ML models served securely using Pickle

---

## 💻 Tech Stack

| Layer        | Technology                |
|--------------|---------------------------|
| Frontend     | Streamlit (Python)        |
| Backend      | Python, Pickle            |
| ML Models    | Random Forest, XGBoost, etc. |
| Deployment   | Streamlit Cloud           |

---

## 🧪 Input Features

### Kidney Disease
- `age`, `bp`, `sg`, `al`, `su`, `rbc`, `pc`, `pcc`, `ba`, `bgr`, `bu`, `sc`, `sod`, `pot`, `hemo`, `pcv`, `wc`, `rc`, `htn`, `dm`, `cad`, `appet`, `pe`, `ane`

### Liver Disease
- `Age`, `Gender`, `Total_Bilirubin`, `Direct_Bilirubin`, `Alkaline_Phosphotase`, `Alamine_Aminotransferase`, `Aspartate_Aminotransferase`, `Total_Protiens`, `Albumin`, `Albumin_and_Globulin_Ratio`

### Parkinson’s Disease
- `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)`, `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP`, `MDVP:Shimmer`, `MDVP:Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA`, `NHR`, `HNR`, `RPDE`, `DFA`, `spread1`, `spread2`, `D2`, `PPE`

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/multi-disease-predictor.git
cd multi-disease-predictor
