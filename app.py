import streamlit as st
import pandas as pandas_lib
import numpy as numpy_lib
from sklearn.linear_model import LogisticRegression

# Page Configuration
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Performance Prediction Dashboard")
st.write("Enter the student's details below to predict their academic performance outcome.")

# Input fields for student metrics
study_hours = st.slider("Weekly Study Hours", 1, 40, 10)
attendance_rate = st.slider("Attendance Rate (%)", 50, 100, 85)
previous_score = st.slider("Previous Exam Score", 0, 100, 70)
extracurricular = st.selectbox("Participates in Extracurriculars?", ["Yes", "No"])

# Convert categorical input to numeric
extracurricular_val = 1 if extracurricular == "Yes" else 0

# Dummy training model setup for demonstration
# (In a real app, you'd load a trained model file using joblib/pickle)
X_train = numpy_lib.array([
    [5, 60, 50, 0],
    [15, 85, 75, 1],
    [25, 95, 90, 1],
    [8, 70, 60, 0],
    [20, 90, 85, 1]
])
y_train = numpy_lib.array([0, 1, 1, 0, 1]) # 1 = Pass/Good, 0 = At Risk

model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction button
if st.button("Predict Performance"):
    user_data = numpy_lib.array([[study_hours, attendance_rate, previous_score, extracurricular_val]])
    prediction = model.predict(user_data)
    
    st.subheader("Results:")
    if prediction[0] == 1:
        st.success("🎉 The student is predicted to **Perform Well / Pass**!")
    else:
        st.warning("⚠️ The student is **At Risk** and may need extra academic support.")
