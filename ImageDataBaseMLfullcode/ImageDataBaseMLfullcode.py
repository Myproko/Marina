import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# ---------- USER CONFIG ----------
DATA_DIR = r"C:\Users\Marina\OneDrive\Desktop\HW\ML\GroupProgect\Depression Data Images\data\train"

SEED = 123
BATCH_SIZE = 32
EPOCHS = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BIG_IMG = (224, 224)
POSITIVE_CLASS = ["Sad","Fear","Angry"]

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------- dataset & dataloaders ----------
def make_dataloaders(data_dir, image_size, batch_size=BATCH_SIZE, val_ratio=0.15, test_ratio=0.15):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    class_names = full_dataset.classes

    total_len = len(full_dataset)
    val_len = int(total_len * val_ratio)
    test_len = int(total_len * test_ratio)
    train_len = total_len - val_len - test_len

    train_ds, val_ds, test_ds = random_split(full_dataset, [train_len, val_len, test_len],
                                             generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_names

train_loader, val_loader, test_loader, class_names = make_dataloaders(DATA_DIR, BIG_IMG)

valid_classes = [cls for cls in POSITIVE_CLASS if cls in class_names]
print("Valid positive classes:", valid_classes)

# ---------- convert multiclass labels -> binary ----------
def convert_to_binary(labels, class_names, positive_class):
    labels_bin = torch.zeros(labels.size(0), dtype=torch.float)
    for cls in positive_class:
        if cls in class_names:
            idx = class_names.index(cls)
            labels_bin += (labels == idx).float()
    labels_bin = torch.clamp(labels_bin, 0, 1)
    return labels_bin.unsqueeze(1)

# ---------- build EfficientNet ----------
def build_efficientnetb0_ft():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False  # Freeze backbone
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 1)
    )
    return model

model = build_efficientnetb0_ft().to(DEVICE)

# ---------- training with early stopping ----------
def train_model_earlystop(model, train_loader, val_loader, epochs=EPOCHS, lr=1e-4,
                          patience=5, factor=0.5, min_lr=1e-6):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor,
                                                     patience=3, min_lr=min_lr)

    best_val_loss = float('inf')
    best_model_state = None
    early_stop_counter = 0

    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels_bin = convert_to_binary(labels, class_names, POSITIVE_CLASS).to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels_bin)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels_bin).sum().item()
            total += labels_bin.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # --- Validation ---
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels_bin = convert_to_binary(labels, class_names, POSITIVE_CLASS).to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels_bin)
                val_loss += loss.item() * imgs.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                correct += (preds == labels_bin).sum().item()
                total += labels_bin.size(0)

        val_loss /= total
        val_acc = correct / total

        # Scheduler
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if early_stop_counter >= patience:
            print(f"⏹ Early stopping triggered after {epoch+1} epochs")
            break

    # Load best model
    model.load_state_dict(best_model_state)
    return model, history

model, history = train_model_earlystop(model, train_loader, val_loader, lr=1e-4, patience=5)

# ---------- evaluation ----------
def evaluate_model(model, test_loader):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    running_loss, correct, total = 0.0, 0, 0
    y_true, y_prob = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            labels_bin = convert_to_binary(labels, class_names, POSITIVE_CLASS).to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels_bin)
            running_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels_bin).sum().item()
            total += labels_bin.size(0)

            y_true.append(labels_bin.cpu().numpy())
            y_prob.append(torch.sigmoid(outputs).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    acc = correct / total
    loss = running_loss / total
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr)

    return loss, acc, auc_score, fpr, tpr, y_true, y_prob

test_res = evaluate_model(model, test_loader)
y_true = test_res[5]
y_prob = test_res[6]
y_pred = (y_prob >= 0.5).astype(int)

# ---------- Confusion Matrix ----------
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=3))

# ---------- ROC Curve ----------
plt.figure(figsize=(8,6))
plt.plot(test_res[3], test_res[4], label=f'Model (AUC={test_res[2]:.3f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# ---------- Training History ----------
def plot_history(hist, title="Model Training"):
    plt.figure(figsize=(12,4))
    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(hist['accuracy'], label='train_acc')
    plt.plot(hist['val_accuracy'], label='val_acc')
    plt.title(title + " - Accuracy")
    plt.legend()
    plt.grid(True)
    # Loss
    plt.subplot(1,2,2)
    plt.plot(hist['loss'], label='train_loss')
    plt.plot(hist['val_loss'], label='val_loss')
    plt.title(title + " - Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_history(history)