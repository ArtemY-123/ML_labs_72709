import os
import time
import random
import psutil
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

random.seed(42)
np.random.seed(42)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

metrics = []
process = psutil.Process(os.getpid())

for name, model in models.items():
    model.fit(X_train, y_train)

    train_times = []
    mem_train_rss = []

    for _ in range(5):
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        train_times.append(time.perf_counter() - start_time)
        mem_train_rss.append(process.memory_info().rss)

    median_train_time = np.median(train_times)
    peak_train_mem_mb = max(mem_train_rss) / (1024 * 1024)

    single_sample = X_test[0].reshape(1, -1)
    inf_times = []
    mem_inf_rss = []

    for _ in range(100):
        start_time = time.perf_counter()
        model.predict(single_sample)
        inf_times.append(time.perf_counter() - start_time)
        mem_inf_rss.append(process.memory_info().rss)

    median_inf_time_ms = np.median(inf_times) * 1000
    peak_inf_mem_mb = max(mem_inf_rss) / (1024 * 1024)

    test_acc = accuracy_score(y_test, model.predict(X_test))

    model_path = f"lab01/results/model_{name}.joblib"
    joblib.dump(model, model_path)
    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024

    metrics.append({
        "Model": name,
        "Accuracy": f"{test_acc:.4f}",
        "Train_Time_s": round(median_train_time, 4),
        "Peak_Mem_MB": round(max(peak_train_mem_mb, peak_inf_mem_mb), 2),
        "Inf_Latency_ms": round(median_inf_time_ms, 4),
        "Size_Bytes": size_bytes,
        "Size_KB": round(size_kb, 2)
    })

df = pd.DataFrame(metrics)

df[['Model', 'Accuracy']].to_csv("lab01/results/baseline_accuracy.csv", index=False)

print("\n--- System Measurements (Copy for report.md) ---")
print(df.to_string(index=False))