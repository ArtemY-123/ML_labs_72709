# Lab 2: A Neural Network from Scratch (Ch. 3)

**Author:** Artem Yuhai  
**Student ID:** 72709  
**Program:** Artificial Intelligence Engineering and Data Science, Vistula University  

## 1. Network Architecture & Mathematical Design
The network is a two-hidden-layer feedforward neural network designed to classify the 64-pixel `load_digits` dataset into 10 classes. 

**Layer Structure:**
*   **Input Layer:** 64 features (scaled by 1/16).
*   **Hidden Layer 1:** Width $W$, ReLU activation. Weight shape: (64, W).
*   **Hidden Layer 2:** Width $W$, ReLU activation. Weight shape: (W, W).
*   **Output Layer:** 10 nodes, Softmax activation. Weight shape: (W, 10).

Weights were initialized from a normal distribution $\mathcal{N}(0, 0.01)$, and biases were initialized to zero. The network was optimized using Stochastic Gradient Descent (SGD) with a mini-batch size of 32 over 50 epochs.

## 2. Experimental Results

The initial cross-entropy loss evaluated to exactly 2.3026, which matches the theoretical expectation of $-\ln(1/10)$ for an untrained network guessing uniformly across 10 classes.

**Training Output:**
```text
Sanity Check: Initial loss is 2.3026 (Expected ~2.3026)
Sanity Check Output Shape: (1257, 10)
Sanity Check Row Sum: 1.0
Gradient Check: 7/10 weights passed relative error < 1e-6
W=16, LR=0.001 | Final Test Acc: 10.19%
W=16, LR=0.01 | Final Test Acc: 10.19%
W=16, LR=0.1 | Final Test Acc: 51.67%
W=64, LR=0.001 | Final Test Acc: 10.19%
W=64, LR=0.01 | Final Test Acc: 11.85%
W=64, LR=0.1 | Final Test Acc: 87.41%