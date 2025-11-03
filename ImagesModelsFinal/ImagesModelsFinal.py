import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POSITIVE_CLASS = "positive_class_name"
class_names = ["class1", "class2", POSITIVE_CLASS]

# --- ??????? ?????? ---
def evaluate_model_progress(model, test_loader, model_name="??????"):
    print(f"\n?? ??????? ?????? {model_name}...")
    model.eval()
    criterion = nn.BCELoss()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true, y_prob = [], []

    for imgs, labels in tqdm(test_loader, desc=f"???????? {model_name}", unit="batch"):
        imgs, labels_bin = imgs.to(DEVICE), (labels==class_names.index(POSITIVE_CLASS)).float().unsqueeze(1).to(DEVICE)
        with torch.no_grad():
            outputs = model(imgs)
            loss = criterion(outputs, labels_bin)

        running_loss += loss.item() * imgs.size(0)
        preds = (outputs>=0.5).float()
        correct += (preds == labels_bin).sum().item()
        total += labels_bin.size(0)
        y_true.append(labels_bin.cpu().numpy())
        y_prob.append(outputs.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    acc = correct / total
    loss = running_loss / total
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr)

    print(f"? {model_name} ??????. Loss: {loss:.4f}, Acc: {acc:.4f}, AUC: {auc_score:.4f}\n")
    return loss, acc, auc_score, fpr, tpr

# --- ??????? ?????????? ---
def train_model_progress(model, train_loader, val_loader, epochs=20, model_name="??????"):
    print(f"\n?? ??????? ?????????? {model_name}...")
    model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"{model_name} Epoch {epoch}/{epochs}", unit="batch")
        for imgs, labels in loop:
            imgs, labels_bin = imgs.to(DEVICE), (labels==class_names.index(POSITIVE_CLASS)).float().unsqueeze(1).to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels_bin)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = (outputs>=0.5).float()
            correct += (preds == labels_bin).sum().item()
            total += labels_bin.size(0)
            loop.set_postfix(loss=running_loss/total, acc=correct/total)

        # ????????? ??????? ????????
        history["train_loss"].append(running_loss/total)
        history["train_acc"].append(correct/total)

        # ?????????
        val_loss, val_acc, _, _, _ = evaluate_model_progress(model, val_loader, model_name + " (Val)")
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    print(f"?? ?????????? {model_name} ?????????.\n")
    return history

# --- ???????? ???? ---
if __name__ == "__main__":
    # ????????????? ???????
    small_cnn = SmallCNN()
    resnet_model = ResNet50Model()
    efficientnet_model = EfficientNetB0Model()

    # DataLoader-?
    train_small = DataLoader(train_dataset_small, batch_size=32, shuffle=True)
    val_small = DataLoader(val_dataset_small, batch_size=32)
    test_small = DataLoader(test_dataset_small, batch_size=32)

    train_big = DataLoader(train_dataset_big, batch_size=32, shuffle=True)
    val_big = DataLoader(val_dataset_big, batch_size=32)
    test_big = DataLoader(test_dataset_big, batch_size=32)

    # --- ?????????? ??????? ---
    hist_small = train_model_progress(small_cnn, train_small, val_small, epochs=20, model_name="Small CNN")
    hist_resnet = train_model_progress(resnet_model, train_big, val_big, epochs=20, model_name="ResNet50")
    hist_efficient = train_model_progress(efficientnet_model, train_big, val_big, epochs=20, model_name="EfficientNetB0")

    # --- ?????? ??????? ?? ????? ---
    _, _, _, fpr_small, tpr_small = evaluate_model_progress(small_cnn, test_small, "Small CNN")
    _, _, _, fpr_resnet, tpr_resnet = evaluate_model_progress(resnet_model, test_big, "ResNet50")
    _, _, _, fpr_efficient, tpr_efficient = evaluate_model_progress(efficientnet_model, test_big, "EfficientNetB0")

    # --- ??????? Loss ? Accuracy ---
    plt.figure(figsize=(12,5))
    for hist, name in zip([hist_small, hist_resnet, hist_efficient],
                          ["Small CNN","ResNet50","EfficientNetB0"]):
        plt.plot(hist["train_loss"], label=f"{name} Train Loss")
        plt.plot(hist["val_loss"], label=f"{name} Val Loss")
    plt.title("Loss ?? ??????")
    plt.xlabel("?????")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12,5))
    for hist, name in zip([hist_small, hist_resnet, hist_efficient],
                          ["Small CNN","ResNet50","EfficientNetB0"]):
        plt.plot(hist["train_acc"], label=f"{name} Train Acc")
        plt.plot(hist["val_acc"], label=f"{name} Val Acc")
    plt.title("Accuracy ?? ??????")
    plt.xlabel("?????")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # --- ROC-?????? ---
    plt.figure(figsize=(8,6))
    plt.plot(fpr_small, tpr_small, label="Small CNN")
    plt.plot(fpr_resnet, tpr_resnet, label="ResNet50")
    plt.plot(fpr_efficient, tpr_efficient, label="EfficientNetB0")
    plt.plot([0,1],[0,1],'k--', label="Random")
    plt.title("ROC-?????? ???????")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()
