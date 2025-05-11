# -*- coding: utf-8 -*-
"""
Created on Sat May  3 16:59:26 2025

@author: Admin
"""

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# ======= LOAD MODEL =======
model = load_model("c:/Users/Admin/Desktop/clg_project/final_mobilenet_model.keras")

# ======= DIRECTORIES =======
train_dir = "C:/Users/Admin/Desktop/clg_project/data/train"
val_dir = "C:/Users/Admin/Desktop/clg_project/data/validation"
test_dir = "C:/Users/Admin/Desktop/clg_project/data/test"

# ======= PARAMETERS =======
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ======= DATA GENERATORS =======
datagen = ImageDataGenerator()

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ======= EVALUATION FUNCTION =======
def evaluate_model(generator, dataset_name):
    print(f"\n📊 Evaluating on {dataset_name} Dataset")

    y_pred = []
    y_true = []

    for images, labels in generator:
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels, axis=1))
        if len(y_true) >= generator.samples:
            break

    y_pred = np.array(y_pred[:generator.samples])
    y_true = np.array(y_true[:generator.samples])

    total = len(y_true)
    correct = np.sum(y_pred == y_true)
    accuracy_manual = (correct / total) * 100

    # Evaluate using model.evaluate
    loss, accuracy_model = model.evaluate(generator, verbose=1)

    print(f"\n✅ Total Samples: {total}")
    print(f"🎯 Correct Predictions: {correct}")
    print(f"📈 Manual Accuracy: {accuracy_manual:.2f}%")
    print(f"📉 Model Evaluate - Loss: {loss:.4f}")
    print(f"📊 Model Evaluate - Accuracy: {accuracy_model:.4f}")

    # Confusion Matrix and Classification Report
    class_labels = list(generator.class_indices.keys())
    print("\n📌 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\n📌 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    # ====== Save Confusion Matrix Heatmap ======
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'{dataset_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'{dataset_name.lower()}_confusion_matrix.png')
    plt.close()

    # ====== Save Precision, Recall, F1-Score Heatmap ======
    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_metrics = df_report.loc[class_labels, ['precision', 'recall', 'f1-score']]

    plt.figure(figsize=(10, 6))
    sns.heatmap(df_metrics, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title(f'{dataset_name} Metrics Heatmap')
    plt.tight_layout()
    plt.savefig(f'{dataset_name.lower()}_metrics_heatmap.png')
    plt.close()

# ======= RUN FOR ALL SETS =======
evaluate_model(train_generator, "Training")
evaluate_model(val_generator, "Validation")
evaluate_model(test_generator, "Test")

import os

# ======= COMBINE HEATMAPS AND MATRICES =======
def combine_images(filenames, output_path):
    images = [Image.open(f) for f in filenames]
    widths, heights = zip(*(i.size for i in images))

    total_height = sum(heights)
    max_width = max(widths)

    combined_img = Image.new('RGB', (max_width, total_height), color=(255, 255, 255))
    
    y_offset = 0
    for im in images:
        combined_img.paste(im, (0, y_offset))
        y_offset += im.size[1]

    combined_img.save(output_path)

    # Delete individual files after combining
    for f in filenames:
        os.remove(f)

# Combine confusion matrices into one image
combine_images(
    ['training_confusion_matrix.png', 'validation_confusion_matrix.png', 'test_confusion_matrix.png'],
    'all_confusion_matrices.png'
)

# Combine precision, recall, F1-score heatmaps into one image
combine_images(
    ['training_metrics_heatmap.png', 'validation_metrics_heatmap.png', 'test_metrics_heatmap.png'],
    'all_metrics_heatmaps.png'
)
