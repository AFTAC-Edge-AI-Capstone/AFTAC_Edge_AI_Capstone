#THIS IS JUST FOR DISPLAY PURPOSES, INFERENCE RAN IN ADVANCE.
#To run inference, please use benchmark-tflite.py in /audio

import streamlit as st
import time
import pandas as pd

data_float16 = {
    "class": ["drone", "jet", "neg", "propPlane", "accuracy", "macro avg", "weighted avg"],
    "precision": ["0.98", "0.93", "0.97", "0.86", "", "0.94", "0.95"],
    "recall": ["1.00", "0.97", "0.98", "0.69", "", "0.91", "0.95"],
    "f1-score": ["0.99", "0.95", "0.98", "0.76", "0.95", "0.92", "0.95"],
    "support": ["148", "392", "487", "120", "1147", "1147", "1147"]
}
df_float16 = pd.DataFrame(data_float16).set_index("class")

data_int8 = {
    "class": ["drone", "jet", "neg", "propPlane", "accuracy", "macro avg", "weighted avg"],
    "precision": ["0.87", "0.83", "0.90", "0.69", "", "0.82", "0.85"],
    "recall": ["0.94", "0.88", "0.89", "0.49", "", "0.80", "0.85"],
    "f1-score": ["0.90", "0.86", "0.90", "0.57", "0.85", "0.81", "0.85"],
    "support": ["148", "392", "487", "120", "1147", "1147", "1147"]
}
df_int8 = pd.DataFrame(data_int8).set_index("class")

@st.fragment
def render():
    st.title("EfficientAT Audio Classifier")
    
    if st.button("Run Benchmark", type="primary"):
        
        with st.spinner("Running benchmark inference"):
            time.sleep(5)
            
        #Float16 Results
        st.subheader("aircraft_mn05_classifier_float16.tflite (2.2 Mb)")
        st.table(df_float16)
        
        st.image("audio/cm_aircraft_mn05_classifier_float16.png")
        
        st.divider() 

        #Int8 Results
        st.subheader("aircraft_mn05_classifier_int8.tflite (1.4 Mb)")
        st.table(df_int8)
        
        st.image("audio/cm_aircraft_mn05_classifier_int8.png")
    
    

   