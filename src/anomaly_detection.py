import pandas as pd
import os
import matplotlib.pyplot as plt


base_dir = os.path.dirname(os.path.abspath(__file__))  
data_path = os.path.join(base_dir, "..", "data", "customer_transaction_data.csv")
output_path = os.path.join(base_dir, "..", "output", "anomalies_detected.csv")


encodings_to_try = ["utf-8", "utf-16", "latin1"]

df = None
for enc in encodings_to_try:
    try:
        df = pd.read_csv(data_path, encoding=enc)
        print(f"✅ Loaded file successfully with encoding: {enc}")
        break
    except Exception as e:
        print(f"Failed with encoding {enc}: {e}")

if df is None:
    raise ValueError("❌ Could not read CSV with any tested encoding.")

print("Columns detected:", df.columns)


mean = df['Sales'].mean()
std = df['Sales'].std()


df['z_score'] = (df['Sales'] - mean) / std


df['is_anomaly'] = df['z_score'].abs() > 3


df[df['is_anomaly']].to_csv(output_path, index=False)


print("✅ Anomalies saved to:", output_path)
print(f"Total transactions: {len(df)}")
print(f"Total anomalies detected: {df['is_anomaly'].sum()}")
print(f"Mean Sales: {mean:.2f}")
print(f"Std Dev Sales: {std:.2f}")
print(f"Anomaly Percentage: {(df['is_anomaly'].sum()/len(df))*100:.2f}%")
print("Preview of anomalies:")
print(df[df['is_anomaly']].head())


plt.figure(figsize=(12,7))
plt.scatter(df.index, df['Sales'], label="Normal", alpha=0.6, color="blue")
plt.scatter(df[df['is_anomaly']].index, df[df['is_anomaly']]['Sales'],
            color="red", label="Anomaly", alpha=0.9, marker="x", s=100)


plt.axhline(mean, color="green", linestyle="--", linewidth=2, label=f"Mean ({mean:.2f})")
plt.axhline(mean + 3*std, color="orange", linestyle="--", linewidth=2, label="+3σ")
plt.axhline(mean - 3*std, color="orange", linestyle="--", linewidth=2, label="-3σ")

plt.xlabel("Transaction Index")
plt.ylabel("Sales")
plt.title("Sales Anomaly Detection with Z-Score Thresholds")
plt.legend()
plt.tight_layout()
plt.show()
