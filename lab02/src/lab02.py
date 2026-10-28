import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Step 1:
# ==========================================
digits = load_digits()
X = digits.data / 16.0
y = digits.target

y_oh = np.eye(10)[y]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_oh, test_size=0.3, stratify=y, random_state=42
)

# Step 2:
# ==========================================
def init_params(W):
    return {
        'W1': np.random.normal(0, 0.01, (64, W)),
        'b1': np.zeros(W),
        'W2': np.random.normal(0, 0.01, (W, W)),
        'b2': np.zeros(W),
        'W3': np.random.normal(0, 0.01, (W, 10)),
        'b3': np.zeros(10)
    }


# Step 3:
# ==========================================
def relu(z):
    return np.maximum(0, z)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward(X, params):
    z1 = X @ params['W1'] + params['b1']
    a1 = relu(z1)

    z2 = a1 @ params['W2'] + params['b2']
    a2 = relu(z2)

    z3 = a2 @ params['W3'] + params['b3']
    a3 = softmax(z3)

    return {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'z3': z3, 'a3': a3}


# Step 4:
# ==========================================
def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m
    return loss


# Step 5:
# ==========================================
def backward(X, y_true, params, cache):
    m = X.shape[0]

    dz3 = cache['a3'] - y_true
    dW3 = (cache['a2'].T @ dz3) / m
    db3 = np.sum(dz3, axis=0) / m

    da2 = dz3 @ params['W3'].T
    dz2 = da2 * (cache['z2'] > 0)
    dW2 = (cache['a1'].T @ dz2) / m
    db2 = np.sum(dz2, axis=0) / m

    da1 = dz2 @ params['W2'].T
    dz1 = da1 * (cache['z1'] > 0)
    dW1 = (X.T @ dz1) / m
    db1 = np.sum(dz1, axis=0) / m

    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}


def gradient_check(X_batch, y_batch, params):
    cache = forward(X_batch, params)
    grads = backward(X_batch, y_batch, params, cache)

    epsilon = 1e-5
    passed = 0

    for _ in range(10):
        i, j = np.random.randint(0, params['W1'].shape[0]), np.random.randint(0, params['W1'].shape[1])

        orig_val = params['W1'][i, j]

        params['W1'][i, j] = orig_val + epsilon
        loss_plus = cross_entropy_loss(forward(X_batch, params)['a3'], y_batch)

        params['W1'][i, j] = orig_val - epsilon
        loss_minus = cross_entropy_loss(forward(X_batch, params)['a3'], y_batch)

        params['W1'][i, j] = orig_val

        num_grad = (loss_plus - loss_minus) / (2 * epsilon)
        analytic_grad = grads['W1'][i, j]

        rel_error = abs(num_grad - analytic_grad) / max(1e-8, abs(num_grad) + abs(analytic_grad))
        if rel_error < 1e-6:
            passed += 1

    print(f"Gradient Check: {passed}/10 weights passed relative error < 1e-6")



# Step 6:
# ==========================================
learning_rates = [0.001, 0.01, 0.1]  # [cite: 1]
widths = [16, 64]  # [cite: 1]
epochs = 50  # [cite: 1]
batch_size = 32  # [cite: 1]

os.makedirs('../results', exist_ok=True)
plt.figure(figsize=(12, 8))

for W in widths:
    for lr in learning_rates:
        params = init_params(W)

        initial_cache = forward(X_train, params)
        initial_loss = cross_entropy_loss(initial_cache['a3'], y_train)
        if W == 16 and lr == 0.001:
            print(f"Sanity Check: Initial loss is {initial_loss:.4f} (Expected ~2.3026)")
            print("Sanity Check Output Shape:", initial_cache['a3'].shape)
            print("Sanity Check Row Sum:", np.sum(initial_cache['a3'][0]))
            gradient_check(X_train[:32], y_train[:32], params)

        history = []

        for epoch in range(epochs):
            indices = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_losses = []

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                cache = forward(X_batch, params)
                loss = cross_entropy_loss(cache['a3'], y_batch)
                epoch_losses.append(loss)

                grads = backward(X_batch, y_batch, params, cache)

                for key in params:
                    params[key] -= lr * grads[key]

            history.append(np.mean(epoch_losses))

        test_cache = forward(X_test, params)
        preds = np.argmax(test_cache['a3'], axis=1)
        trues = np.argmax(y_test, axis=1)
        acc = np.mean(preds == trues)
        print(f"W={W}, LR={lr} | Final Test Acc: {acc * 100:.2f}%")

        plt.plot(history, label=f'W={W}, LR={lr}')

plt.title('Training Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.legend()
plt.grid(True)
plt.savefig('results/loss_curves.png')
plt.close()


# Step 7:
# ==========================================
times_fwd = []
times_bwd = []
params = init_params(64)
X_batch = X_train[:32]
y_batch = y_train[:32]

for _ in range(100):
    t0 = time.perf_counter()
    cache = forward(X_batch, params)
    t1 = time.perf_counter()
    backward(X_batch, y_batch, params, cache)
    t2 = time.perf_counter()

    times_fwd.append(t1 - t0)
    times_bwd.append(t2 - t1)

print(f"\nTiming (Median of 100 runs):")  # [cite: 1]
print(f"Forward Pass:  {np.median(times_fwd) * 1000:.4f} ms")
print(f"Backward Pass: {np.median(times_bwd) * 1000:.4f} ms")