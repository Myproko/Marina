# -*- coding: utf-8 -*-
"""
Binary detection of 'Sad' class using:
  - custom small CNN (input 18x18)
  - ResNet50 (transfer learning, input 224x224)
  - EfficientNetB0 (transfer learning, input 224x224)

Dataset layout (example):
data/
  Angry/
  Disgust/
  Fear/
  Happy/
  Neutral/
  Sad/
  Surprize/

Author: ChatGPT (example)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model

# ---------- USER CONFIG ----------
DATA_DIR = r"C:\Users\Marina\OneDrive\Desktop\HW\ML\GroupProgect\Depression Data Images\data"

# general
SEED = 123
BATCH_SIZE = 32
EPOCHS = 20     # можно увеличить до 50, если ресурсы позволяют
AUTOTUNE = tf.data.AUTOTUNE

# sizes
SMALL_IMG = (18, 18)   # для собственной CNN
BIG_IMG = (224, 224)   # для ResNet/EfficientNet
# ---------------------------------

# проверка пути
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"DATA_DIR не найден: {DATA_DIR}")

# ---------- helper: load dataset (train/val/test) ----------
def make_tf_datasets(data_dir, image_size, batch_size=BATCH_SIZE, val_ratio=0.15, test_ratio=0.15, seed=SEED):
    """
    Загружает датасет из директории, возвращает (train_ds, val_ds, test_ds, class_names)
    Использует validation_split на image_dataset_from_directory, затем делит валид на вал+тест.
    """
    # validation_split param expects fraction of total dataset for validation. We'll set validation_split = val+test
    val_plus_test = val_ratio + test_ratio
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        validation_split=val_plus_test,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )

    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        validation_split=val_plus_test,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )

    # Split val_test_ds into val_ds and test_ds (by batches)
    val_test_count = tf.data.experimental.cardinality(val_test_ds).numpy()
    # Если мало батчей, делим по батчам (приблизительно)
    val_batches = val_test_count // 2 if val_test_count >= 2 else 1
    val_ds = val_test_ds.take(val_batches)
    test_ds = val_test_ds.skip(val_batches)

    class_names = train_ds.class_names
    return train_ds, val_ds, test_ds, class_names

# ---------- convert multiclass labels -> binary (Sad vs notSad) ----------
def to_binary_label_dataset(ds, class_names, positive_class='Sad'):
    """Преобразует датасет с метками 0..K-1 в (images, label_binary) где label_binary = 1 если класс == positive_class"""
    pos_idx = class_names.index(positive_class)
    def map_fn(images, labels):
        labels_bin = tf.cast(tf.equal(labels, pos_idx), tf.float32)  # 1.0 for Sad, 0.0 for others
        return images, labels_bin
    return ds.map(map_fn, num_parallel_calls=AUTOTUNE)

# ---------- preprocessing pipelines ----------
def prepare_for_training(ds, rescale=True, augment=False):
    """Применяет нормализацию, кеширование, shuffle, prefetch."""
    if rescale:
        normalizer = tf.keras.layers.Rescaling(1./255)
        ds = ds.map(lambda x, y: (normalizer(x), y), num_parallel_calls=AUTOTUNE)
    if augment:
        # простая аугментация (можно добавить больше)
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.05),
        ])
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    return ds

# ---------- build models ----------
def build_small_cnn(input_shape=(18,18,3)):
    """Простая CNN с тремя сверточными слоями -> для 18x18 изображений"""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs, name='small_cnn_3conv')
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', keras.metrics.AUC(name='auc')])
    return model

def build_resnet50(input_shape=(224,224,3), base_trainable=False):
    """ResNet50 transfer learning for binary classification"""
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = base_trainable
    inputs = layers.Input(shape=input_shape)
    x = layers.Lambda(preprocess_input)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs, name='resnet50_ft')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy', keras.metrics.AUC(name='auc')])
    return model

def build_efficientnetb0(input_shape=(224,224,3), base_trainable=False):
    """EfficientNetB0 transfer learning for binary classification"""
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications.efficientnet import preprocess_input
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = base_trainable
    inputs = layers.Input(shape=input_shape)
    x = layers.Lambda(preprocess_input)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs, name='efficientnetb0_ft')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy', keras.metrics.AUC(name='auc')])
    return model

# ---------- prepare datasets ----------
# 1) datasets for small model (no resizing needed)
train_small, val_small, test_small, class_names = make_tf_datasets(DATA_DIR, image_size=SMALL_IMG, batch_size=BATCH_SIZE)
# 2) datasets for big models (resized)
train_big, val_big, test_big, _ = make_tf_datasets(DATA_DIR, image_size=BIG_IMG, batch_size=BATCH_SIZE)

# Convert to binary (Sad vs not-Sad)
train_small_bin = to_binary_label_dataset(train_small, class_names, positive_class='Sad')
val_small_bin   = to_binary_label_dataset(val_small, class_names, positive_class='Sad')
test_small_bin  = to_binary_label_dataset(test_small, class_names, positive_class='Sad')

train_big_bin = to_binary_label_dataset(train_big, class_names, positive_class='Sad')
val_big_bin   = to_binary_label_dataset(val_big, class_names, positive_class='Sad')
test_big_bin  = to_binary_label_dataset(test_big, class_names, positive_class='Sad')

# Apply preprocessing (normalization, augmentation only on train)
train_small_bin = prepare_for_training(train_small_bin, rescale=True, augment=True)
val_small_bin = prepare_for_training(val_small_bin, rescale=True, augment=False)
test_small_bin = prepare_for_training(test_small_bin, rescale=True, augment=False)

train_big_bin = prepare_for_training(train_big_bin, rescale=False, augment=True)  # rescaling will be done by preprocess_input in models (we used Lambda preprocess_input)
val_big_bin = prepare_for_training(val_big_bin, rescale=False, augment=False)
test_big_bin = prepare_for_training(test_big_bin, rescale=False, augment=False)

# ---------- build models ----------
small_cnn = build_small_cnn(input_shape=(SMALL_IMG[0], SMALL_IMG[1], 3))
resnet_model = build_resnet50(input_shape=(BIG_IMG[0], BIG_IMG[1], 3), base_trainable=False)
efficientnet_model = build_efficientnetb0(input_shape=(BIG_IMG[0], BIG_IMG[1], 3), base_trainable=False)

# печать сводки
print("Классы:", class_names)
print("\n--- Small CNN ---")
small_cnn.summary()
print("\n--- ResNet50 top ---")
resnet_model.summary()
print("\n--- EfficientNetB0 top ---")
efficientnet_model.summary()

# ---------- callbacks ----------
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
]

# ---------- TRAINING ----------
history_small = small_cnn.fit(
    train_small_bin,
    validation_data=val_small_bin,
    epochs=EPOCHS,
    callbacks=callbacks
)

history_resnet = resnet_model.fit(
    train_big_bin,
    validation_data=val_big_bin,
    epochs=EPOCHS,
    callbacks=callbacks
)

history_efficient = efficientnet_model.fit(
    train_big_bin,
    validation_data=val_big_bin,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ---------- EVALUATION ----------
test_res_small = small_cnn.evaluate(test_small_bin, verbose=2)
test_res_resnet = resnet_model.evaluate(test_big_bin, verbose=2)
test_res_efficient = efficientnet_model.evaluate(test_big_bin, verbose=2)

print("\nTest (Small CNN)  -> Loss: {:.4f}  Acc: {:.4f}  AUC: {:.4f}".format(*test_res_small))
print("Test (ResNet50)   -> Loss: {:.4f}  Acc: {:.4f}  AUC: {:.4f}".format(*test_res_resnet))
print("Test (EffNetB0)   -> Loss: {:.4f}  Acc: {:.4f}  AUC: {:.4f}".format(*test_res_efficient))

# ---------- PLOTTING: training curves ----------
def plot_history(hist, title):
    plt.figure(figsize=(12,4))
    # accuracy
    plt.subplot(1,2,1)
    plt.plot(hist.history['accuracy'], label='train_acc')
    plt.plot(hist.history['val_accuracy'], label='val_acc')
    plt.title(title + " — Accuracy")
    plt.legend()
    plt.grid(True)
    # loss
    plt.subplot(1,2,2)
    plt.plot(hist.history['loss'], label='train_loss')
    plt.plot(hist.history['val_loss'], label='val_loss')
    plt.title(title + " — Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_history(history_small, "Small CNN (18x18)")
plot_history(history_resnet, "ResNet50 (224x224)")
plot_history(history_efficient, "EfficientNetB0 (224x224)")

# ---------- Compare final test metrics in bar chart ----------
models_names = ['Small CNN', 'ResNet50', 'EfficientNetB0']
test_accs = [test_res_small[1], test_res_resnet[1], test_res_efficient[1]]
test_aucs = [test_res_small[2], test_res_resnet[2], test_res_efficient[2]]

x = np.arange(len(models_names))
width = 0.35
plt.figure(figsize=(10,4))
plt.bar(x - width/2, test_accs, width, label='Accuracy')
plt.bar(x + width/2, test_aucs, width, label='AUC')
plt.xticks(x, models_names)
plt.ylim(0,1.0)
plt.ylabel('Score')
plt.title('Comparison on test set')
plt.legend()
plt.grid(axis='y')
plt.show()

# ---------- Optional: ROC curves ----------
from sklearn.metrics import roc_curve, auc

def get_probs_and_labels(model, ds):
    y_true = []
    y_prob = []
    for x_batch, y_batch in ds.unbatch().batch(1024):
        preds = model.predict(x_batch, verbose=0)
        y_prob.append(preds.ravel())
        y_true.append(y_batch.numpy().ravel())
    y_prob = np.concatenate(y_prob)
    y_true = np.concatenate(y_true)
    return y_true, y_prob

# Берём небольшую выборку test (unbatch->batch) чтобы вычислить ROC
y_true_s, y_prob_s = get_probs_and_labels(small_cnn, test_small_bin)
y_true_r, y_prob_r = get_probs_and_labels(resnet_model, test_big_bin)
y_true_e, y_prob_e = get_probs_and_labels(efficientnet_model, test_big_bin)

fpr_s, tpr_s, _ = roc_curve(y_true_s, y_prob_s)
fpr_r, tpr_r, _ = roc_curve(y_true_r, y_prob_r)
fpr_e, tpr_e, _ = roc_curve(y_true_e, y_prob_e)

auc_s = auc(fpr_s, tpr_s)
auc_r = auc(fpr_r, tpr_r)
auc_e = auc(fpr_e, tpr_e)

plt.figure(figsize=(8,6))
plt.plot(fpr_s, tpr_s, label=f'Small CNN (AUC={auc_s:.3f})')
plt.plot(fpr_r, tpr_r, label=f'ResNet50 (AUC={auc_r:.3f})')
plt.plot(fpr_e, tpr_e, label=f'EfficientNetB0 (AUC={auc_e:.3f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves (Sad vs not-Sad)')
plt.legend()
plt.grid(True)
plt.show()

# ---------- Show some test images with predictions ----------
def show_predictions(model, ds, num_images=9):
    ds_iter = ds.unbatch().batch(32)
    batch = next(iter(ds_iter))
    imgs, labels = batch
    preds = model.predict(imgs)
    plt.figure(figsize=(8,8))
    for i in range(min(num_images, imgs.shape[0])):
        ax = plt.subplot(3,3,i+1)
        plt.imshow((imgs[i].numpy() * 255).astype('uint8'))
        true = 'Sad' if labels.numpy()[i] == 1 else 'Not-Sad'
        pred_prob = float(preds[i].ravel()[0])
        pred_label = 'Sad' if pred_prob >= 0.5 else 'Not-Sad'
        plt.title(f"True:{true}\nPred:{pred_label} ({pred_prob:.2f})")
        plt.axis('off')
    plt.show()

print("Пример предсказаний Small CNN (на тесте):")
show_predictions(small_cnn, test_small_bin)

print("Пример предсказаний ResNet50 (на тесте):")
show_predictions(resnet_model, test_big_bin)

print("Пример предсказаний EfficientNetB0 (на тесте):")
show_predictions(efficientnet_model, test_big_bin)

