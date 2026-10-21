# Lab 3: Architectures and their costs - Report

## 1. Hand-Calculated vs. Measured FLOPs
*(Note: Replace the hand-calculated values below with your specific math from Chapter 4 if required; these are the baseline calculations based on the provided architectures).*

* **MLP:** The measured FLOPs (469,504) closely align with the theoretical calculation of $2 \times N_{in} \times N_{out}$ for each linear layer. The first layer dominates the cost ($2 \times 784 \times 256 = 401,408$ FLOPs).
* **CNN:** The measured FLOPs are significantly higher (2,063,488) despite having the fewest parameters (20,490). This is because the convolution formula ($2 \times K^2 \times C_{in} \times C_{out} \times H_{out} \times W_{out}$) applies the weights across the entire spatial dimension of the image, leading to massive computational reuse.
* **Attention:** The FLOPs (1,182,976) sit between the MLP and CNN. The cost is driven by the QKV projections ($2 \times seq \times d\_model \times 3 \times d\_model$) and the attention score calculations ($2 \times seq^2 \times d\_model$), where $seq = 16$.

## 2. Architecture Trade-offs (Accuracy, Parameters, Latency)
Based on the generated plots:
* **Accuracy vs. Parameters:** The CNN is highly parameter-efficient, achieving the highest accuracy (89.59%) with the fewest parameters (~20k)[cite: 2]. The MLP requires over 10x the parameters (~235k) to achieve a slightly lower accuracy (87.85%)[cite: 2]. The Attention model underperformed in both metrics (83.13% accuracy, ~70k params)[cite: 2], likely because Transformers lack inductive bias (like translation invariance in CNNs) and typically require much larger datasets or pre-training to excel.
* **Accuracy vs. Latency:** The MLP is the fastest (0.79 ms for batch 128), while the CNN is the slowest (16.05 ms for batch 128)[cite: 2]. The CNN's latency penalty is due to the dense computational requirements of sliding the convolutional kernels across the feature maps.

## 3. Cost-Effectiveness and Access Patterns
Using the access patterns described in Reddi Ch. 4.6-4.7:
* **Most Cost-Effective for Compute-Constrained Edge Devices:** The **MLP** is highly effective if latency is the primary constraint. Linear layers are memory-bandwidth bound (they require loading large weight matrices once per pass), but because the CPU/hardware doesn't have to perform massive FLOPs, it executes extremely fast (0.79 ms). 
* **Most Cost-Effective for Accuracy/Memory-Constrained Devices:** The **CNN** is the clear winner for parameter efficiency and accuracy. However, convolutions are compute-bound with high data reuse (the same small $3\times3$ kernel weights are read repeatedly to multiply against different input pixels). This high arithmetic intensity leads to the highest FLOPs and the highest latency (16.05 ms), making it costly in terms of power and time, but highly efficient in terms of storage footprint.