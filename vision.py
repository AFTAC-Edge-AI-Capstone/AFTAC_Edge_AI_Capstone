import streamlit as st
import re

table_html = """
<style>
    .v-table { width: 100%; border-collapse: collapse; background-color: #1a1a1a; color: #efefef; }
    .v-table th, .v-table td { border: 1px solid #444; padding: 10px; font-size: 14px; }
    .main-h { background-color: #4a0e0e; text-align: center; font-weight: bold; }
    .sub-h { background-color: #2d0a0a; color: #ff9999; font-weight: bold; }
    .red-cell { background-color: #631212; color: white; text-align: center; font-weight: bold; }
    .white-cell { background-color: #f0f0f0; color: black; text-align: center; font-weight: bold; }
</style>

<table class="v-table">
    <tr class="main-h"><td colspan="4">VISION MODEL — QUANTIZATION IMPACT SUMMARY</td></tr>
    <tr>
        <th>METRIC</th>
        <th>PRE-QUANTIZED ONNX</th>
        <th>LOCALLY QUANTIZED ONNX</th>
        <th>DIFFERENCE</th>
    </tr>
    <tr class="sub-h"><td colspan="4">HOST-SIDE VALIDATION AND COMPRESSION</td></tr>
    <tr><td>Model format</td><td>FP32 ONNX</td><td>INT8/QDQ ONNX</td><td>Quantized</td></tr>
    <tr><td>Top-1 Accuracy</td><td>76.09%</td><td>76.01%</td><td class="red-cell">-0.08 pp</td></tr>
    <tr><td>Top-5 Accuracy</td><td>92.93%</td><td>92.96%</td><td class="white-cell">+0.03 pp</td></tr>
    <tr><td>Throughput Speedup</td><td>47.10 img/s</td><td>134.65 img/s</td><td class="white-cell">185.88%</td></tr>
    <tr class="sub-h"><td colspan="4">TARGET EDGE DEPLOYMENT METRICS</td></tr>
    <tr><td>Deployable on STM32N6570-DK</td><td class="red-cell">No</td><td colspan="2" class="white-cell">Yes</td></tr>
    <tr><td>Board latency / inference</td><td>N/A</td><td colspan="2">49.08 ms</td></tr>
    <tr><td>Board throughput</td><td>N/A</td><td colspan="2">20.37 inf/s</td></tr>
    <tr class="sub-h"><td colspan="4">CSWIN-TINY FOOTPRINT SUMMARY</td></tr>
    <tr><td>Top-1 Accuracy</td><td>82.82%</td><td>Weights footprint</td><td>85.148 MB</td></tr>
    <tr><td>Top-5 Accuracy</td><td>96.30%</td><td>ONNX File Size</td><td>87,461,000 B</td></tr>
</table>
"""

@st.fragment
def render():
    st.title("The Computer Vision Model")
    
    st.markdown(re.sub(r'\n\s*\n', '\n', table_html), unsafe_allow_html=True)