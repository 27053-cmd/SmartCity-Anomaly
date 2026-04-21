import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns   # for nicer plots
import streamlit as st


base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "..", "data", "customer_transaction_data.csv")
output_path = os.path.join(base_dir, "..", "output", "anomalies_detected.csv")


df = pd.read_csv(data_path)


mean = df['Sales'].mean()
std = df['Sales'].std()
df['z_score'] = (df['Sales'] - mean) / std
df['is_anomaly'] = df['z_score'].abs() > 3


df[df['is_anomaly']].to_csv(output_path, index=False)


st.set_page_config(page_title="Smart City Dashboard", layout="wide")

st.markdown("<h1 style='text-align: center; color: #2E86C1;'>📊 Smart City Sales Anomaly Dashboard</h1>", unsafe_allow_html=True)


st.subheader("Dataset Preview (Scrollable)")
st.dataframe(df.head(50), height=300)


st.subheader("Summary Statistics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Transactions", len(df))
col2.metric("Total Anomalies", df['is_anomaly'].sum())
col3.metric("Mean Sales", f"{mean:.2f}")
col4.metric("Std Dev Sales", f"{std:.2f}")
col5.metric("Anomaly %", f"{(df['is_anomaly'].sum()/len(df))*100:.2f}%")


st.subheader("Anomalies Detected (Scrollable)")
st.dataframe(df[df['is_anomaly']], height=300)


st.subheader("Sales Anomaly Detection Visualization")
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(12,6))
ax.scatter(df.index, df['Sales'], label="Normal", alpha=0.6, color="#3498DB")
ax.scatter(df[df['is_anomaly']].index, df[df['is_anomaly']]['Sales'],
           color="#E74C3C", label="Anomaly", alpha=0.9, marker="x", s=100)
ax.axhline(mean, color="#27AE60", linestyle="--", linewidth=2, label=f"Mean ({mean:.2f})")
ax.axhline(mean + 3*std, color="#F39C12", linestyle="--", linewidth=2, label="+3σ")
ax.axhline(mean - 3*std, color="#F39C12", linestyle="--", linewidth=2, label="-3σ")
ax.set_xlabel("Transaction Index")
ax.set_ylabel("Sales")
ax.set_title("Sales Anomaly Detection", fontsize=16, fontweight="bold")
ax.legend()
st.pyplot(fig)


st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Smart City Project • Powered by Streamlit</p>", unsafe_allow_html=True)
