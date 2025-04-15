import streamlit as st
import joblib
import numpy as np

# --- Custom CSS styling ---
st.markdown("""
    <style>
        

        h2 {
            color: #007BFF;
            font-size: 36px;
            text-align: center;
        }

        p {
            font-size: 16px;
            color: #333;
        }

        /* Agrandir les champs input */
        input {
            font-size: 18px !important;
        }

        /* Styliser le selectbox */
        .stSelectbox label {
            font-size: 22px !important;
            color: #8B0000 !important;
            font-weight: bold;
        }

        /* Bouton Predict */
        .stButton > button {
            background-color: #32CD32;
            color: white;
            font-size: 20px;
            border-radius: 10px;
            padding: 10px 20px;
        }

        .stButton > button:hover {
            background-color: #109B10;
            color: #FFFFFF;
            transform-scale: 0.1
        }
    </style>
""", unsafe_allow_html=True)

# --- Load models and scalers ---
model_red = joblib.load('winequality-red.joblib')
scaler_red = joblib.load('scaler-red.joblib')

model_white = joblib.load('winequality-white.joblib')
scaler_white = joblib.load('scaler-white.joblib')

# --- Feature Ranges ---
feature_ranges = {
    "Red Wine": {
        "Fixed Acidity": (4.6, 15.9, 8.3),
        "Volatile Acidity": (0.12, 1.58, 0.53),
        "Citric Acid": (0.0, 1.0, 0.27),
        "Residual Sugar": (0.9, 15.5, 2.5),
        "Chlorides": (0.012, 0.611, 0.09),
        "Free Sulfur Dioxide": (1.0, 72.0, 15.0),
        "Total Sulfur Dioxide": (6.0, 289.0, 46.0),
        "Density": (0.99007, 1.00369, 0.9967),
        "pH": (2.74, 4.01, 3.31),
        "Sulphates": (0.33, 2.0, 0.66),
        "Alcohol": (8.4, 14.9, 10.4)
    },
    "White Wine": {
        "Fixed Acidity": (3.8, 14.2, 6.9),
        "Volatile Acidity": (0.08, 1.10, 0.27),
        "Citric Acid": (0.0, 1.66, 0.33),
        "Residual Sugar": (0.6, 65.8, 6.4),
        "Chlorides": (0.009, 0.346, 0.045),
        "Free Sulfur Dioxide": (2.0, 289.0, 35.0),
        "Total Sulfur Dioxide": (9.0, 440.0, 138.0),
        "Density": (0.98711, 1.03898, 0.9940),
        "pH": (2.72, 3.82, 3.2),
        "Sulphates": (0.22, 1.08, 0.49),
        "Alcohol": (8.0, 14.2, 10.5)
    }
}

# --- Section sélection du type de vin ---
st.markdown("""
    <h2 style='text-align: center;color: #8B0000; margin-bottom: 10px'>
        🍷 <strong>Wine Quality Prediction</strong>
    </h2>
""", unsafe_allow_html=True)
st.write("")

st.markdown("""
    <h4 style='color: #007BFF; '>
        <strong>Select the Type of Wine</strong>
    </h4>
""", unsafe_allow_html=True)

wine_type = st.selectbox("", ("Red Wine", "White Wine"))

# --- Mise en forme dynamique ---
if wine_type == "Red Wine":
    model = model_red
    scaler = scaler_red
    
else:
    model = model_white
    scaler = scaler_white
    

st.markdown("""
    <h4 style='color: #007BFF; '>
        <strong>Enter the wine characteristics below to predict its quality</strong>
    </h4>
""", unsafe_allow_html=True)
st.write("")

# --- Input des caractéristiques ---
r = feature_ranges[wine_type]

feature1 = st.number_input("Fixed Acidity", *r["Fixed Acidity"])
feature2 = st.number_input("Volatile Acidity", *r["Volatile Acidity"])
feature3 = st.number_input("Citric Acid", *r["Citric Acid"])
feature4 = st.number_input("Residual Sugar", *r["Residual Sugar"])
feature5 = st.number_input("Chlorides", *r["Chlorides"])
feature6 = st.number_input("Free Sulfur Dioxide", *r["Free Sulfur Dioxide"])
feature7 = st.number_input("Total Sulfur Dioxide", *r["Total Sulfur Dioxide"])
feature8 = st.number_input("Density", *r["Density"])
feature9 = st.number_input("pH", *r["pH"])
feature10 = st.number_input("Sulphates", *r["Sulphates"])
feature11 = st.number_input("Alcohol", *r["Alcohol"])

input_data = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, 
                        feature7, feature8, feature9, feature10, feature11]])
input_data = scaler.transform(input_data)

# --- Prédiction ---
if st.button('Predict'):
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)

    st.write("### 🎯 Predicted Class:")
    if prediction[0] == 0:
        st.success("Low Quality 🍷")
    elif prediction[0] == 1:
        st.info("Medium Quality 🍷🍷")
    elif prediction[0] == 2:
        st.success("High Quality 🍷🍷🍷")
    else:
        st.warning("Unknown Class ❓")

    st.write("### 📊 Prediction Probabilities:")
    st.write(f"Low Quality (0): {proba[0][0]*100:.2f}%")
    st.write(f"Medium Quality (1): {proba[0][1]*100:.2f}%")
    st.write(f"High Quality (2): {proba[0][2]*100:.2f}%")

# --- Sidebar ---
# --- Sidebar dynamique ---
st.sidebar.header("Model Details")

if wine_type == "Red Wine":
    st.sidebar.write("""
    This Machine Learning model predicts the quality of **red wine** based on its physicochemical properties.  
    It was trained on the **winequality-red.csv** dataset from **Kaggle**, which includes features like acidity, alcohol content, pH, and more.

    The model is built using an ensemble approach called **Voting Classifier**, combining two different algorithms:  
    - **K-Nearest Neighbors (KNN)**: captures local similarity between data points.  
    - **Random Forest Classifier**: handles complex feature interactions and reduces overfitting.

    A **Grid Search** was used to optimize the hyperparameters for both algorithms.  
    The final Voting Classifier was trained using the best-found parameters, achieving an accuracy of **89.20%** on the test set.

    Enter your values in the fields to predict the wine’s quality instantly.
    """)
else:
    st.sidebar.write("""
    This Machine Learning model predicts the quality of **white wine** based on its physicochemical properties.  
    It was trained on the **winequality-white.csv** dataset from **Kaggle**, including features like acidity, sugar content, density, and more.

    The model is built using an ensemble approach called **Voting Classifier**, combining two different algorithms:  
    - **K-Nearest Neighbors (KNN)**: captures local similarity between data points.  
    - **Random Forest Classifier**: handles complex feature interactions and reduces overfitting.

    Hyperparameters were optimized using **Grid Search**.  
    The model achieved an accuracy of **87.62%** on the test set.

    Enter your values in the fields to predict the wine’s quality instantly.
    """)

