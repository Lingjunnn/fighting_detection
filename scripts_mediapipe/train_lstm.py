import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight


# ---------------------------
# Dataset
# ---------------------------
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------
# LSTM Model
# ---------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim=66, hidden_dim=256, num_layers=3, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        o, _ = self.lstm(x)
        out = o[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)



# ---------------------------
# Train Function
# ---------------------------
def train(model, loader, criterion, optimizer, device):
    model.train()
    total, correct, total_loss = 0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        total += X.size(0)
        correct += (pred.argmax(1) == y).sum().item()

    return total_loss / total, correct / total


# ---------------------------
# Eval Function
# ---------------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, total_loss = 0, 0, 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)

            total_loss += loss.item() * X.size(0)
            total += X.size(0)
            correct += (pred.argmax(1) == y).sum().item()

    return total_loss / total, correct / total


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../data/datasets/sequences")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    # Load data
    X_train = np.load(f"{args.data_path}/X_train.npy")
    y_train = np.load(f"{args.data_path}/y_train.npy")
    X_val = np.load(f"{args.data_path}/X_val.npy")
    y_val = np.load(f"{args.data_path}/y_val.npy")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Dataset & DataLoader
    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(SeqDataset(X_val, y_val), batch_size=args.batch)

    # Model
    model = LSTMModel().to(device)

    # Class weights
    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train
    )
    cw = torch.tensor(cw, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_acc = 0

    # Training Loop
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] Train: loss={train_loss:.4f}, acc={train_acc:.4f} | Val: loss={val_loss:.4f}, acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "../models/best_model.pth")
            print("  -> Saved new best model")

    print("Training complete!")
    print("Best val accuracy:", best_acc)


if __name__ == "__main__":
    main()
