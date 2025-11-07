import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, matthews_corrcoef,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
tf.debugging.set_log_device_placement(False)

# Display settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
# Dataset path
DATA_DIR = ('https://www.kaggle.com/datasets/khairunneesa/depression-dataset-on-facial-ecpression-images/data')

# Image parameters
IMG_SIZE = 224  # Resize from 524x524 to 224x224
BATCH_SIZE = 32
NUM_CLASSES = 9
EPOCHS = 50

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

print(f"Configuration:")
print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Split: Train={TRAIN_RATIO:.0%}, Val={VAL_RATIO:.0%}, Test={TEST_RATIO:.0%}")
# Load dataset
data_dir = Path(data_dir)
paths, labels, classes = [], [], []
    
    for folder in sorted(data_dir.iterdir()):
        if folder.is_dir():
            classes.append(folder.name)
            for img in folder.glob('*.*'):
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    paths.append(str(img))
                    labels.append(folder.name)
    
    return paths, labels, classes

print("Loading dataset...")
all_paths, all_labels, class_names = load_dataset(DATA_DIR)

print(f"Dataset loaded successfully!")
print(f"Total images: {len(all_paths)}")
print(f"Number of classes: {len(class_names)}")
print(f"Classes: {class_names}")
# Analyze class distribution
counts = Counter(all_labels)
df_dist = pd.DataFrame([
    {'Class': c, 'Count': n, 'Percentage': n/len(all_labels)*100}
    for c, n in counts.most_common()
])

print("\nClass Distribution:")
print(df_dist.to_string(index=False))

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(df_dist['Class'], df_dist['Count'], color='steelblue')
ax1.set_xlabel('Class')
ax1.set_ylabel('Count')
ax1.set_title('Class Distribution - Counts')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

ax2.pie(df_dist['Count'], labels=df_dist['Class'], autopct='%1.1f%%')
ax2.set_title('Class Distribution - Percentages')

plt.tight_layout()
plt.show()

# Check imbalance
ratio = df_dist['Count'].max() / df_dist['Count'].min()
print(f"Imbalance ratio: {ratio:.2f}x")

if ratio > 2.0:
    print("Dataset is imbalanced...Stratification is necessary")
    print("We'll use Focal Loss and class weights.")
else:
    print("Dataset is relatively balanced.")
# Step 1: Split off test set (15%)
train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    all_paths,
    all_labels,
    test_size=TEST_RATIO,
    stratify=all_labels,
    random_state=42
)

# Step 2: Split train/val (70%/15%)
# To get 70% train and 15% val from original: 15/(70+15) = 0.1765
val_size_adj = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_val_paths,
    train_val_labels,
    test_size=val_size_adj,
    stratify=train_val_labels,
    random_state=42
)

print("Data Split Summary:")
print(f"  Total: {len(all_paths)} images")
print(f"  Train: {len(train_paths)} ({len(train_paths)/len(all_paths)*100:.1f}%)")
print(f"  Val: {len(val_paths)} ({len(val_paths)/len(all_paths)*100:.1f}%)")
print(f"  Test: {len(test_paths)} ({len(test_paths)/len(all_paths)*100:.1f}%)")
# Get distribution for each split
def get_dist(labels):
    counts = Counter(labels)
    total = len(labels)
    return {c: (n/total)*100 for c, n in counts.items()}

dists = {
    'Full Dataset': get_dist(all_labels),
    'Train': get_dist(train_labels),
    'Validation': get_dist(val_labels),
    'Test': get_dist(test_labels)
}

# Create comparison table
rows = []
for c in sorted(class_names):
    rows.append({
        'Class': c,
        'Full%': f"{dists['Full Dataset'][c]:.2f}",
        'Train%': f"{dists['Train'][c]:.2f}",
        'Val%': f"{dists['Validation'][c]:.2f}",
        'Test%': f"{dists['Test'][c]:.2f}"
    })

print("Stratification Verification:")
print(pd.DataFrame(rows).to_string(index=False))

# Visual verification
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Class Distribution - Stratification Verification', fontsize=16)

colors = ['gray', 'steelblue', 'orange', 'green']
for ax, (name, dist), color in zip(axes.flat, dists.items(), colors):
    ax.bar(class_names, [dist[c] for c in class_names], color=color, alpha=0.7)
    ax.set_title(name)
    ax.set_ylabel('Percentage (%)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
# Compute class weights
label_to_idx = {l: i for i, l in enumerate(sorted(set(all_labels)))}
train_labels_int = [label_to_idx[l] for l in train_labels]

weights_array = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels_int),
    y=train_labels_int
)

class_weights = {i: w for i, w in enumerate(weights_array)}

print("Class Weights:")
for label, idx in sorted(label_to_idx.items()):
    print(f"  {label}: {class_weights[idx]:.3f}")
    # Data Generators
    #def create_datagen():
   # """ Create ImageDataGenerator with augmentation for training and rescaling for val/test. """
   class FocalLoss(keras.losses.Loss):
    """
    Focal Loss for addressing class imbalance.
    
    FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    
    Args:
        alpha: Weighting factor in [0, 1] (default: 0.25)
        gamma: Focusing parameter >= 0 (default: 2.0)
    """
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate cross-entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calculate focal weight
        focal_weight = self.alpha * tf.pow(1 - y_pred, self.gamma)
        
        # Calculate focal loss
        focal_loss = focal_weight * cross_entropy
        
        return tf.reduce_sum(focal_loss, axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({'alpha': self.alpha, 'gamma': self.gamma})
        return config
# Create data generators
# Training set augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

# Val/Test: no augmentation (only rescaling)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create DataFrames
train_df = pd.DataFrame({'path': train_paths, 'label': train_labels})
val_df = pd.DataFrame({'path': val_paths, 'label': val_labels})
test_df = pd.DataFrame({'path': test_paths, 'label': test_labels})

# Create generators
train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col='path',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

val_gen = val_test_datagen.flow_from_dataframe(
    val_df,
    x_col='path',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=42
)

test_gen = val_test_datagen.flow_from_dataframe(
    test_df,
    x_col='path',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=42
)

print(f"  Train batches: {len(train_gen)}")
print(f"  Val batches: {len(val_gen)}")
print(f"  Test batches: {len(test_gen)}")
print(f"  Class indices: {train_gen.class_indices}")
def create_simple_cnn(lr=0.001, dropout=0.3, use_focal=False):
    """
    Simple CNN with 3 convolutional layers.
    """
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', 
                     input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(2),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    loss = FocalLoss() if use_focal else 'categorical_crossentropy'
    loss_name = "Focal Loss" if use_focal else "Categorical Cross-Entropy"
    
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=loss,33
        metrics=['accuracy']
    )
    
    print(f"  Loss function: {loss_name}")
    return model
def create_resnet(freeze=True, lr=0.0001, weights=None):
    """
    ResNet50 with possibility of using transfer learning.
    """
    base = ResNet50(
        include_top=False,
        weights=weights,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    if freeze:
        # Freeze all layers except last 20
        for layer in base.layers[:-20]:
            layer.trainable = False
    
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"  Trainable parameters: {trainable:,}")
    
    return model
# Training callbacks
def get_callbacks():
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

# Storage for results
results = {}
histories = {}
# Experiment 1: Simple CNN (Baseline)
print("="*60)
print("EXPERIMENT 1: Simple CNN - Baseline")
print("="*60)

model_exp1 = create_simple_cnn(lr=0.001, use_focal=False)

hist_exp1 = model_exp1.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=get_callbacks(),
    verbose=1
)

histories['Exp1_SimpleCNN_Baseline'] = hist_exp1
results['Exp1_SimpleCNN_Baseline'] = model_exp1

print("Training for Experiment 1 completed!")
# Experiment 2: Simple CNN with Focal Loss
# Experiment 3: ResNet50 Train from Scratch
print("="*60)
print("EXPERIMENT 3: ResNet50 - Train from Scratch")
print("="*60)

model_exp3 = create_resnet(freeze=False, lr=0.001, weights=None)

hist_exp3 = model_exp3.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=get_callbacks(),
    verbose=1
)

histories['Exp3_ResNet'] = hist_exp3
results['Exp3_ResNet'] = model_exp3

print("Training for Experiment 3 completed!")

# Experiment 4: ResNet50 Transfer Learning
print("="*60)
print("EXPERIMENT 4: ResNet50 - Transfer Learning")
print("="*60)

model_exp4 = create_resnet(freeze=True, lr=0.0001, weights='imagenet')

hist_exp4 = model_exp4.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=get_callbacks(),
    verbose=1
)

histories['Exp4_ResNet_Transfer'] = hist_exp4
results['Exp4_ResNet_Transfer'] = model_exp4

print("Training for Experiment 4 completed!")
# Experiment 5: ResNet50 Fine-tuning
print("="*60)
print("EXPERIMENT 5: ResNet50 - Fine-tuning")
print("="*60)

model_exp5 = create_resnet(freeze=False, lr=0.0001, weights='imagenet')

hist_exp5 = model_exp5.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=get_callbacks(),
    verbose=1
)

histories['Exp5_ResNet_Finetune'] = hist_exp5
results['Exp5_ResNet_Finetune'] = model_exp5

print("Training for Experiment 5 completed!")
print("="*60)
print("ALL EXPERIMENTS COMPLETED!")
print("="*60)
# Plot training curves
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle('Training Curves - All Experiments', fontsize=16)

for idx, (name, hist) in enumerate(histories.items()):
    ax = axes[idx//2, idx%2]
    ax.plot(hist.history['accuracy'], 'o-', label='Train')
    ax.plot(hist.history['val_accuracy'], 's-', label='Val')
    ax.set_title(name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
def evaluate_with_roc(model, gen, name="Test"):
    """
    Evaluation with ROC/AUC.
    """
    print(f"\nEvaluating on {name} set...")
    gen.reset()
    
    # Get predictions
    y_pred_proba = model.predict(gen, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = gen.classes
    
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'cm': confusion_matrix(y_true, y_pred)
    }
    
    # ROC/AUC - Multi-class (one-vs-rest)
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    
    fpr, tpr, roc_auc = {}, {}, {}
    
    # Per-class ROC
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true_bin.ravel(),
        y_pred_proba.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Macro-average AUC
    roc_auc["macro"] = np.mean([roc_auc[i] for i in range(NUM_CLASSES)])
    
    metrics.update({'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc})
    return metrics
# Evaluate all models on validation and test sets
val_metrics = {}
test_metrics = {}

print("="*80)
print("EVALUATING ALL MODELS")
print("="*80)

for name, model in results.items():
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    
    val_metrics[name] = evaluate_with_roc(model, val_gen, "Validation")
    test_metrics[name] = evaluate_with_roc(model, test_gen, "Test")
    
    print(f"Results:")
    print(f"  Val Accuracy: {val_metrics[name]['accuracy']:.4f}")
    print(f"  Test Accuracy: {test_metrics[name]['accuracy']:.4f}")
    print(f"  Test AUC (macro): {test_metrics[name]['roc_auc']['macro']:.4f}")
    # Create results comparison table
comp_data = []
for name in results.keys():
    comp_data.append({
        'Experiment': name,
        'Val_Acc': val_metrics[name]['accuracy'],
        'Test_Acc': test_metrics[name]['accuracy'],
        'Test_Prec': test_metrics[name]['precision'],
        'Test_Rec': test_metrics[name]['recall'],
        'Test_F1': test_metrics[name]['f1'],
        'Test_MCC': test_metrics[name]['mcc'],
        'Test_AUC': test_metrics[name]['roc_auc']['macro']
    })

results_df = pd.DataFrame(comp_data)

print("="*100)
print("RESULTS COMPARISON")
print("="*100)
print(results_df.to_string(index=False))
# Visualize ROC curves for the best model on test set
def plot_roc_curves(metrics, exp_name):
    """
    Plot ROC curves.
    """
    fpr, tpr, roc_auc = metrics['fpr'], metrics['tpr'], metrics['roc_auc']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'ROC Curves - {exp_name}', fontsize=16)
    
    # Plot 1: All classes
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
    
    for i, color in zip(range(NUM_CLASSES), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'Class {i} (AUC={roc_auc[i]:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC=0.5)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - All Classes')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)
    
    # Plot 2: Micro-average
    ax = axes[1]
    ax.plot(fpr["micro"], tpr["micro"], 'b-', lw=3,
            label=f'Micro-avg (AUC={roc_auc["micro"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Micro-Average ROC\nMacro AUC={roc_auc["macro"]:.3f}')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

# Plot ROC curves for all experiments
for name, metrics in test_metrics.items():
    plot_roc_curves(metrics, name)

# Rank models by different metrics
print("\n" + "="*80)
print("MODEL RANKINGS ON TEST SET")
print("="*80)

print("Ranked by Test Accuracy:")
for idx, row in results_df.sort_values('Test_Acc', ascending=False).iterrows():
    print(f"  {idx+1}. {row['Experiment']}: {row['Test_Acc']:.4f}")

print("Ranked by Test AUC (Macro):")
for idx, row in results_df.sort_values('Test_AUC', ascending=False).iterrows():
    print(f"  {idx+1}. {row['Experiment']}: {row['Test_AUC']:.4f}")

print("Ranked by Test F1-Score:")
for idx, row in results_df.sort_values('Test_F1', ascending=False).iterrows():
    print(f"  {idx+1}. {row['Experiment']}: {row['Test_F1']:.4f}")

# Best overall model (weighted score)
results_df['Overall_Score'] = (
    results_df['Test_Acc'] * 0.3 +
    results_df['Test_F1'] * 0.3 +
    results_df['Test_AUC'] * 0.4
)

best = results_df.loc[results_df['Overall_Score'].idxmax()]

print("\n" + "="*80)
print("RECOMMENDED MODEL")
print("="*80)
print(f"\n{best['Experiment']}")
print(f"\n  Test Performance:")
print(f"    Accuracy: {best['Test_Acc']:.4f}")
print(f"    Precision: {best['Test_Prec']:.4f}")
print(f"    Recall: {best['Test_Rec']:.4f}")
print(f"    F1-Score: {best['Test_F1']:.4f}")
print(f"    MCC: {best['Test_MCC']:.4f}")
print(f"    AUC (macro): {best['Test_AUC']:.4f}")


