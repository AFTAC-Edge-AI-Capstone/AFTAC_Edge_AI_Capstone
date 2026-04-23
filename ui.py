import streamlit as st
import maintenance
import audio1
import audio2
import vision

# --- STREAMLIT APP ---
st.set_page_config(page_title="AFTAC Edge AI", layout="centered")

st.markdown("""
        <style>
        .block-container {
            padding-top: 3.75rem !important;
        }
        </style>
""", unsafe_allow_html=True)

page_map = {
    "Maintenance": ("build", maintenance),
    "Audio 1": ("volume_up", audio1),
    "Audio 2": ("volume_up", audio2),
    "Vision": ("visibility", vision),
}

with st.container(horizontal_alignment='center'):
    selected_page = st.segmented_control(
        label="Navigation",
        label_visibility="collapsed",
        options=page_map.keys(),
        format_func=lambda x: f":material/{page_map[x][0]}: {x}",
        default='Maintenance',
        required=True
    )

page_map[selected_page][1].render()