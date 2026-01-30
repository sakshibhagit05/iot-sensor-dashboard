import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(layout="wide", page_title="IoT Sensor Dashboard")

# -----------------------------
# Title
# -----------------------------
st.title("📡 Smart IoT Sensor Data Visualization Dashboard")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Dashboard Controls")

np.random.seed(42)
n = st.sidebar.slider("Number of sensor readings", 50, 500, 100)

chart_type = st.sidebar.selectbox(
    "Choose Visualization",
    [
        "Line Trend",
        "Histogram",
        "Scatter",
        "Box Plot",
        "Bar Average",
        "Correlation Heatmap",
        "PCA Plot"
    ]
)

# -----------------------------
# Data Generation
# -----------------------------
data = pd.DataFrame({
    "Temperature": np.random.normal(25, 3, n),
    "Humidity": np.random.normal(60, 10, n),
    "Pressure": np.random.normal(1013, 5, n),
    "Air_Quality": np.random.normal(300, 50, n)
})

# -----------------------------
# Top Metrics Row
# -----------------------------
st.subheader("📊 Sensor Summary Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Avg Temp", round(data["Temperature"].mean(), 2))
col2.metric("Avg Humidity", round(data["Humidity"].mean(), 2))
col3.metric("Avg Pressure", round(data["Pressure"].mean(), 2))
col4.metric("Avg Air Quality", round(data["Air_Quality"].mean(), 2))

# -----------------------------
# Data Table Expandable
# -----------------------------
with st.expander("📋 View Raw Sensor Data"):
    st.dataframe(data)

# -----------------------------
# CSV Download
# -----------------------------
st.download_button(
    "⬇️ Download CSV",
    data.to_csv(index=False),
    file_name="sensor_data.csv"
)

st.markdown("---")
st.header("📈 Visualization Output")

# -----------------------------
# Chart Logic
# -----------------------------
if chart_type == "Line Trend":
    fig, ax = plt.subplots()
    ax.plot(data["Temperature"], label="Temperature")
    ax.plot(data["Humidity"], label="Humidity")
    ax.plot(data["Pressure"], label="Pressure")
    ax.legend()
    ax.set_title("Sensor Trend Over Time")
    st.pyplot(fig, use_container_width=True)

elif chart_type == "Histogram":
    fig, ax = plt.subplots()
    sns.histplot(data["Temperature"], bins=20, ax=ax)
    ax.set_title("Temperature Distribution")
    st.pyplot(fig, use_container_width=True)

elif chart_type == "Scatter":
    fig, ax = plt.subplots()
    ax.scatter(data["Temperature"], data["Humidity"])
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Humidity")
    ax.set_title("Temperature vs Humidity")
    st.pyplot(fig, use_container_width=True)

elif chart_type == "Box Plot":
    fig, ax = plt.subplots()
    sns.boxplot(data=data, ax=ax)
    ax.set_title("Sensor Spread (Box Plot)")
    st.pyplot(fig, use_container_width=True)

elif chart_type == "Bar Average":
    fig, ax = plt.subplots()
    data.mean().plot(kind="bar", ax=ax)
    ax.set_title("Average Sensor Values")
    st.pyplot(fig, use_container_width=True)

elif chart_type == "Correlation Heatmap":
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig, use_container_width=True)

elif chart_type == "PCA Plot":
    features = data[["Temperature", "Humidity", "Pressure", "Air_Quality"]]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    p = pca.fit_transform(scaled)

    fig, ax = plt.subplots()
    ax.scatter(p[:, 0], p[:, 1])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Visualization")
    st.pyplot(fig, use_container_width=True)

    st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)
