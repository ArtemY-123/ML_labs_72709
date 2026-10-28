import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import time
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
from thop import profile

# 1. Dataset Preparation
def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    train_dataset, val_dataset = random_split(full_train_dataset, [55000, 5000])
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    return train_loader, val_loader, test_loader

# 2. Define Models
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        return self.net(x)


class AttentionNet(nn.Module):
    def __init__(self):
        super(AttentionNet, self).__init__()
        self.patch_size = 7
        self.num_patches = 16
        self.patch_dim = 1 * 7 * 7

        self.proj = nn.Linear(self.patch_dim, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, batch_first=True, activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        B = x.size(0)
        x = x.unfold(2, 7, 7).unfold(3, 7, 7)
        x = x.contiguous().view(B, 16, 49)

        x = self.proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

# 3. Cost Computation & Latency
def compute_costs_and_latency(model, model_name, device):
    print(f"\n--- Profiling {model_name} ---")
    model.to(device)
    model.eval()

    try:
        summary(model, input_size=(1, 1, 28, 28), device=device)
    except NameError:
        pass

    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    try:
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops = macs * 2
        print(f"[{model_name}] THOP FLOPs: {flops:,.0f} | Params: {params:,.0f}")
    except NameError:
        pass

    def measure_latency(batch_size, runs):
        latencies = []
        dummy_batch = torch.randn(batch_size, 1, 28, 28).to(device)

        for _ in range(5):
            _ = model(dummy_batch)

        with torch.no_grad():
            for _ in range(runs):
                start = time.perf_counter()
                _ = model(dummy_batch)
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
        return np.median(latencies)

    latency_b1 = measure_latency(batch_size=1, runs=100)
    latency_b128 = measure_latency(batch_size=128, runs=20)

    print(f"[{model_name}] Latency (Batch=1): {latency_b1:.2f} ms")
    print(f"[{model_name}] Latency (Batch=128): {latency_b128:.2f} ms")

    return params, latency_b128

# 4. Training Loop
def train_and_evaluate(model, train_loader, test_loader, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 5

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# 5. Execution and Plotting
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders()

    models = {
        "MLP": MLP(),
        "CNN": CNN(),
        "Attention": AttentionNet()
    }

    results = {}

    for name, model in models.items():
        params, lat = compute_costs_and_latency(model, name, device)
        print(f"Training {name} for 5 epochs...")
        acc = train_and_evaluate(model, train_loader, test_loader, device)
        print(f"[{name}] Test Accuracy: {acc:.2f}%\n")

        results[name] = {
            "params": params,
            "latency": lat,
            "accuracy": acc
        }

    names = list(results.keys())
    accuracies = [results[n]["accuracy"] for n in names]
    params_counts = [results[n]["params"] / 1000 for n in names]
    latencies = [results[n]["latency"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for i, txt in enumerate(names):
        ax1.scatter(params_counts[i], accuracies[i], label=txt, s=100)
        ax1.annotate(txt, (params_counts[i], accuracies[i]), xytext=(5, 5), textcoords='offset points')
    ax1.set_xlabel('Parameters (Thousands)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy vs Parameters')
    ax1.grid(True)

    for i, txt in enumerate(names):
        ax2.scatter(latencies[i], accuracies[i], label=txt, s=100)
        ax2.annotate(txt, (latencies[i], accuracies[i]), xytext=(5, 5), textcoords='offset points')
    ax2.set_xlabel('Latency for Batch 128 (ms)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy vs Latency')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('architecture_tradeoffs.png')
    plt.show()
    print("Saved plot to 'architecture_tradeoffs.png'.")