import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import roc_curve, auc

# ---------- USER CONFIG ----------
DATA_DIR = r"C:\Users\Marina\OneDrive\Desktop\HW\ML\GroupProgect\Depression Data Images\data\train"

SEED = 123
BATCH_SIZE = 32
EPOCHS = 59

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SMALL_IMG = (18, 18)
BIG_IMG = (224, 224)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------- dataset & dataloaders ----------
def make_dataloaders(data_dir, image_size, batch_size=BATCH_SIZE, val_ratio=0.15, test_ratio=0.15):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
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

train_small, val_small, test_small, class_names = make_dataloaders(DATA_DIR, SMALL_IMG)
train_big, val_big, test_big, _ = make_dataloaders(DATA_DIR, BIG_IMG)

POSITIVE_CLASS = ["Sad","Fear","Angry"]
valid_classes = [cls for cls in POSITIVE_CLASS if cls in class_names]
print("Valid positive classes:", valid_classes)

# ---------- convert multiclass labels -> binary ----------
def convert_to_binary(labels, class_names, positive_class):
    labels_bin = torch.zeros(labels.size(0), dtype=torch.float)
    for cls in positive_class:
        if cls in class_names:
            idx = class_names.index(cls)
            labels_bin += (labels == idx).float()
    labels_bin = torch.clamp(labels_bin, 0, 1)  # ensure binary 0/1
    return labels_bin.unsqueeze(1)

# ---------- augmentation & normalization ----------
train_small.transform = transforms.Compose([
    transforms.Resize(SMALL_IMG),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.RandomResizedCrop(SMALL_IMG),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_big.transform = transforms.Compose([
    transforms.Resize(BIG_IMG),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.RandomResizedCrop(BIG_IMG),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---------- build models ----------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64,1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def build_resnet50_ft():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = True
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256,1)
    )
    return model

def build_efficientnetb0_ft():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = True
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256,1)
    )
    return model

# ---------- select model here ----------
#model = SmallCNN().to(DEVICE)
#model = build_resnet50_ft().to(DEVICE)
model = build_efficientnetb0_ft().to(DEVICE)

# ---------- training utils with Early Stopping ----------
def train_model_earlystop(model, train_loader, val_loader, epochs=EPOCHS, lr=1e-4,
                          patience=7, factor=0.5, min_lr=1e-6):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor,
                                                     patience=3, min_lr=min_lr)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_loss = float('inf')
    best_model_state = None
    early_stop_counter = 0

    for epoch in range(epochs):
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

        train_losses.append(running_loss / total)
        train_accs.append(correct / total)

        # validation
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

        val_loss = val_loss / total
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Scheduler step
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_losses[-1]:.4f} Acc: {train_accs[-1]:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if early_stop_counter >= patience:
            print(f"⏹ Early stopping triggered after {epoch+1} epochs")
            break

    # load best model
    model.load_state_dict(best_model_state)

    return {"loss": train_losses, "val_loss": val_losses, "accuracy": train_accs, "val_accuracy": val_accs}

# ---------- train model ----------
train_loader, val_loader = train_big, val_big  # Use BIG_IMG loaders for ResNet/EfficientNet
history = train_model_earlystop(model, train_loader, val_loader, lr=1e-4,
                                patience=10, factor=0.5, min_lr=1e-6)

# ---------- evaluation ----------
def evaluate_model(model, test_loader):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    correct = 0
    total = 0
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
    return loss, acc, auc_score, fpr, tpr

test_loader = test_big
test_res = evaluate_model(model, test_loader)

print("\nTest Results  -> Loss: {:.4f}  Acc: {:.4f}  AUC: {:.4f}".format(*test_res[:3]))

# ---------- plotting ----------
def plot_history(hist, title):
    plt.figure(figsize=(12,4))
    # accuracy
    plt.subplot(1,2,1)
    plt.plot(hist['accuracy'], label='train_acc')
    plt.plot(hist['val_accuracy'], label='val_acc')
    plt.title(title + " - Accuracy")
    plt.legend()
    plt.grid(True)
    # loss
    plt.subplot(1,2,2)
    plt.plot(hist['loss'], label='train_loss')
    plt.plot(hist['val_loss'], label='val_loss')
    plt.title(title + " - Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_history(history, "Model Training")

# ---------- ROC curve ----------
plt.figure(figsize=(8,6))
plt.plot(test_res[3], test_res[4], label=f'Model (AUC={test_res[2]:.3f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()