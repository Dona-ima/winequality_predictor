import streamlit as st
import joblib
import numpy as np

# Custom CSS styling
st.markdown("""
    <style>
        h1 {
            color: #007BFF;
            font-size: 36px;
            text-align: center;
        }
        .css-18e3th9 {
            background-color: #f2f2f2;
            padding: 20px;
        }
        p {
            font-size: 16px;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

# Load the model and scaler
model = joblib.load('winequality-red.joblib')
scaler = joblib.load('scaler.joblib')

st.title("Red Wine Quality Prediction")
st.write("Enter the wine characteristics below to predict its quality:")

# Input features
feature1 = st.number_input("Fixed Acidity", min_value=4.6, max_value=15.9, step=0.1, value=8.3)
feature2 = st.number_input("Volatile Acidity", min_value=0.12, max_value=1.58, step=0.01, value=0.53)
feature3 = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, step=0.01, value=0.27)
feature4 = st.number_input("Residual Sugar", min_value=0.9, max_value=15.5, step=0.1, value=2.5)
feature5 = st.number_input("Chlorides", min_value=0.012, max_value=0.611, step=0.001, value=0.09)
feature6 = st.number_input("Free Sulfur Dioxide", min_value=1.0, max_value=72.0, step=1.0, value=15.9)
feature7 = st.number_input("Total Sulfur Dioxide", min_value=6.0, max_value=289.0, step=1.0, value=46.5)
feature8 = st.number_input("Density", min_value=0.99007, max_value=1.00369, step=0.00001, value=0.9967)
feature9 = st.number_input("pH", min_value=2.74, max_value=4.01, step=0.01, value=3.31)
feature10 = st.number_input("Sulphates", min_value=0.33, max_value=2.0, step=0.01, value=0.66)
feature11 = st.number_input("Alcohol", min_value=8.4, max_value=14.9, step=0.1, value=10.4)


input_data = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, 
                        feature7, feature8, feature9, feature10, feature11]])
input_data = scaler.transform(input_data)

# Predict
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

    # Show prediction probabilities
    st.write("### 📊 Prediction Probabilities:")
    st.write(f"Low Quality (0): {proba[0][0]*100:.2f}%")
    st.write(f"Medium Quality (1): {proba[0][1]*100:.2f}%")
    st.write(f"High Quality (2): {proba[0][2]*100:.2f}%")

# Sidebar
st.sidebar.header("Model Details")
st.sidebar.write("""
This Machine Learning model predicts the quality of red wine based on its physicochemical properties.  
It was trained on the **winequality-red.csv** dataset from **Kaggle**, which includes features like acidity, alcohol content, pH, and more.

The model is built using an ensemble approach called **Voting Classifier**, combining two different algorithms:  
- **K-Nearest Neighbors (KNN)**: captures local similarity between data points.  
- **Random Forest Classifier**: handles complex feature interactions and reduces overfitting.

A **Grid Search** was used to optimize the hyperparameters for both algorithms.  
The final Voting Classifier was trained using the best-found parameters, achieving an accuracy of **84.55%** on the test set.

Enter your values in the fields to predict the wine’s quality instantly.
""")
