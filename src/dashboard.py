import pandas as pd
import os
import matplotlib.pyplot as plt
import streamlit as st

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "..", "data", "customer_transaction_data.csv")
output_path = os.path.join(base_dir, "..", "output", "anomalies_detected.csv")

# Load data
df = pd.read_csv(data_path)

# Calculate stats
mean = df['Sales'].mean()
std = df['Sales'].std()
df['z_score'] = (df['Sales'] - mean) / std
df['is_anomaly'] = df['z_score'].abs() > 3

# Save anomalies
df[df['is_anomaly']].to_csv(output_path, index=False)

# Streamlit UI
st.title("📊 Smart City Sales Anomaly Dashboard")

st.write("### Dataset Preview (Scrollable)")
# Show first 50 rows, keep table scrollable in fixed height
st.dataframe(df.head(50), height=300)

st.write("### Summary Statistics")
st.metric("Total Transactions", len(df))
st.metric("Total Anomalies", df['is_anomaly'].sum())
st.metric("Mean Sales", f"{mean:.2f}")
st.metric("Std Dev Sales", f"{std:.2f}")
st.metric("Anomaly %", f"{(df['is_anomaly'].sum()/len(df))*100:.2f}%")

st.write("### Anomalies Detected (Scrollable)")
st.dataframe(df[df['is_anomaly']], height=300)

# Visualization
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(df.index, df['Sales'], label="Normal", alpha=0.6, color="blue")
ax.scatter(df[df['is_anomaly']].index, df[df['is_anomaly']]['Sales'],
           color="red", label="Anomaly", alpha=0.9, marker="x", s=100)
ax.axhline(mean, color="green", linestyle="--", label=f"Mean ({mean:.2f})")
ax.axhline(mean + 3*std, color="orange", linestyle="--", label="+3σ")
ax.axhline(mean - 3*std, color="orange", linestyle="--", label="-3σ")
ax.set_xlabel("Transaction Index")
ax.set_ylabel("Sales")
ax.set_title("Sales Anomaly Detection")
ax.legend()
st.pyplot(fig)
