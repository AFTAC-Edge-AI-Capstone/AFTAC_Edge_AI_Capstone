import streamlit as st

@st.fragment
def render():
    st.title("The Predictive Maintenance Model")
    
    st.subheader("EfficientAT + Classifier Quantized to float16")
    st.image("audio/cm_aircraft_mn05_classifier_float16.png")

    st.subheader("EfficientAT + Classifier Quantized to int8")
    st.image("audio/cm_aircraft_mn05_classifier_int8.png")