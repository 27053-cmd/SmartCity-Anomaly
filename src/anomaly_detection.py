import pandas as pd
import os
import matplotlib.pyplot as plt

# Use relative paths so the script works regardless of username or machine
base_dir = os.path.dirname(os.path.abspath(__file__))   # points to src/
data_path = os.path.join(base_dir, "..", "data", "customer_transaction_data.csv")
output_path = os.path.join(base_dir, "..", "output", "anomalies_detected.csv")

# Try multiple encodings to avoid Unicode errors
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

# Step 2: Calculate mean and std for 'Sales' column
mean = df['Sales'].mean()
std = df['Sales'].std()

# Step 3: Compute Z-Score
df['z_score'] = (df['Sales'] - mean) / std

# Step 4: Flag anomalies
df['is_anomaly'] = df['z_score'].abs() > 3

# Step 5: Save anomalies to output folder
df[df['is_anomaly']].to_csv(output_path, index=False)

# Step 6: Print summary
print("✅ Anomalies saved to:", output_path)
print(f"Total transactions: {len(df)}")
print(f"Total anomalies detected: {df['is_anomaly'].sum()}")
print(f"Mean Sales: {mean:.2f}")
print(f"Std Dev Sales: {std:.2f}")
print(f"Anomaly Percentage: {(df['is_anomaly'].sum()/len(df))*100:.2f}%")
print("Preview of anomalies:")
print(df[df['is_anomaly']].head())

# Step 7: Visualization
plt.figure(figsize=(12,7))
plt.scatter(df.index, df['Sales'], label="Normal", alpha=0.6, color="blue")
plt.scatter(df[df['is_anomaly']].index, df[df['is_anomaly']]['Sales'],
            color="red", label="Anomaly", alpha=0.9, marker="x", s=100)

# Add mean and threshold lines
plt.axhline(mean, color="green", linestyle="--", linewidth=2, label=f"Mean ({mean:.2f})")
plt.axhline(mean + 3*std, color="orange", linestyle="--", linewidth=2, label="+3σ")
plt.axhline(mean - 3*std, color="orange", linestyle="--", linewidth=2, label="-3σ")

plt.xlabel("Transaction Index")
plt.ylabel("Sales")
plt.title("Sales Anomaly Detection with Z-Score Thresholds")
plt.legend()
plt.tight_layout()
plt.show()
