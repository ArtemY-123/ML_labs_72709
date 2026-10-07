### 1. Goal
The objective of this lab is to establish a reproducible machine learning systems environment and measure the system cost—specifically training time, inference latency, model size, and peak memory—of baseline linear and ensemble models to evaluate their suitability for various hardware deployment environments.

### 2. Method
An isolated Python 3.11.8 virtual environment was configured with strictly pinned dependencies. The Breast Cancer Wisconsin dataset was loaded and split 70/30 (stratified, random_state=42). We trained a `LogisticRegression` (max_iter=1000) and a `RandomForestClassifier` (n_estimators=100). System costs were measured following a strict methodology: training time is the median of 5 runs after a 1-run warm-up, inference latency is the median of 100 single-sample predictions, model size was captured from the serialized `joblib` file, and peak memory was tracked via `psutil`.

### 3. Results

| Model | Accuracy | Train_Time_s | Peak_Mem_MB | Inf_Latency_ms | Size_Bytes | Size_KB |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LogisticRegression | 0.9415 | 0.8399 | 139.76 | 0.1863 | 1055 | 1.03 |
| RandomForest | 0.9357 | 0.3446 | 140.91 | 6.3907 | 290889 | 284.07 |

### 4. Conclusions

* **Performance vs. Efficiency Trade-offs:** The Logistic Regression model slightly outperformed the Random Forest in test accuracy (0.9415 vs 0.9357) while simultaneously being significantly more efficient. It is roughly 275 times smaller in disk size (1.03 KB vs 284.07 KB) and over 34 times faster during single-sample inference (0.1863 ms vs 6.3907 ms).
* **The Memory Bottleneck:** Despite both models boasting sub-10 ms inference latencies that easily qualify for TinyML constraints, the Python runtime and data processing overhead caused peak memory consumption to reach ~140 MB. This memory footprint completely disqualifies both models from TinyML (≤ 256 KB limit) and pushes them into larger resource tiers. 
* **Deployment Suitability:** Based on the established system budgets, both models are suited for **Mobile** deployment. The models fit perfectly within the Mobile constraints: their peak memory of ~140 MB fits the 64-256 MB budget, their latencies are well below the 20 ms limit, and their disk sizes are drastically smaller than the 10 MB maximum. They could also be deployed on Edge or Cloud environments, but Mobile is the smallest, most restrictive budget category they successfully satisfy.