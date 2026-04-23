import streamlit as st
import pandas as pd
import plotly.express as px
from sensor_health.config import DATASETS, WINDOW, MAX_RUL
import sensor_health.utils
import sensor_health.tflite

def get_data():
    train_X, val_X, test_X, train_y, val_y, test_y, num_features = sensor_health.utils.load_data(DATASETS, WINDOW, MAX_RUL, "sensor_health/data")
    return test_X, test_y

def predict(data):
    interpreter = sensor_health.tflite.load_model("sensor_health/models/maintenance_model_int8.tflite")
    return sensor_health.tflite.tflite_predict(interpreter, data)

@st.fragment
def render():
    st.title("The Predictive Maintenance Model")

    if 'predictions' not in st.session_state:
        st.session_state.predictions = None

    X_test, y_test = get_data()

    if st.session_state.predictions is None:
        with st.container(horizontal_alignment="center"):
            st.markdown("""
                <style>
                    .muted-heading {
                        text-align: center;
                        opacity: 0.7;
                    }
                    .muted-heading svg {
                        display: none;
                    }
                </style>
                <h3 class="muted-heading">Please predict to begin</h3>
            """, unsafe_allow_html=True)
            if st.button("Run predictions on test data"):
                with st.spinner("Inference in progress..."):
                    st.session_state.predictions = predict(X_test)
                    st.rerun(scope='fragment')
                st.success("Done!")
    else:
        # --- SECTION 1: Bulk Predictions ---
        st.header("1. Test Data Overview")
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            "Sample Index": list(range(len(y_test))),
            "Actual": y_test * MAX_RUL,
            "Predicted": st.session_state.predictions * MAX_RUL
        })
        
        # Scatter plot: Actual vs Predicted
        fig_scatter = px.scatter(
            plot_df, x="Actual", y="Predicted", 
            hover_data=["Sample Index"],
            title="Remaining Useful Life (RUL): Actual vs. Predicted",
            labels={"Actual": "Actual RUL", "Predicted": "Predicted RUL"}
        )
        # Add a diagonal line for reference
        fig_scatter.add_shape(
            type="line", line=dict(dash="dash"),
            x0=plot_df['Actual'].min(), y0=plot_df['Actual'].min(), 
            x1=plot_df['Actual'].max(), y1=plot_df['Actual'].max()
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # --- SECTION 2: Individual Sample Inspection ---
        render_detailed(X_test, y_test)

@st.fragment
def render_detailed(X_test, y_test):
        st.divider()
        st.header("2. Detailed Sample Inspection")

        selected_idx = st.selectbox(
            "Select a test sample to examine:", 
            options=list(range(len(X_test))),
            format_func=lambda x: f"Sample {x}"
        )

        # Layout for sample details
        col1, col2 = st.columns([1, 3])

        with col1:
            st.metric("Actual RUL", f"{y_test[selected_idx]*MAX_RUL:.4f}")
            current_pred = st.session_state.predictions[selected_idx]
            st.metric("Model Prediction", f"{current_pred*MAX_RUL:.4f}")

        with col2:
            # Prepare sensor data for plotting
            # X_test[idx] is list[list[float]] -> [timestep][sensor]
            sample_data = X_test[selected_idx]
            df_sensors = pd.DataFrame(
                sample_data, 
                columns=[f"Sensor {i}" for i in range(len(sample_data[0]))]
            )
            df_sensors.index.name = "Timestep"
            
            fig_sensors = px.line(
                df_sensors, 
                title=f"Sensor Readings Over Time (Sample {selected_idx})",
                labels={"value": "Reading", "variable": "Sensor"}
            )
            st.plotly_chart(fig_sensors, use_container_width=True)