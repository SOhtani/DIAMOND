import os
import random
import numpy as np
import pandas as pd
from pathlib import Path 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import Callback 
from sklearn.metrics import auc, confusion_matrix, roc_curve
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import csv
from scipy.ndimage import zoom

input_dir = '/Users/jykim/Desktop/DIAMOND/DIAMOND/Data'
output_dir_base = '/Users/jykim/Desktop/DIAMOND/DIAMOND/output'

# 手順 2: データを読み込み、前処理を行う
def load_dataset():
    airleak_data, no_complication_data = [], []
    airleak_files, no_complication_files = [], []
    airleak_dates, no_complication_dates = [], []
    for root, _, files in os.walk(output_dir_base):
        if "data until 24h" in root:
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path, sep=';', parse_dates=['Date'])
                    if df.shape[0] < 120:
                        print(f"Skipping {file} with {df.shape[0]} rows")
                        continue
                    data = df.loc[:, ['Air leak']].values  # Edit this line
                    initial_date = df['Date'].iloc[0]  
                    if "no-complication" in root:
                        no_complication_data.append(data)
                        no_complication_files.append(file)
                        no_complication_dates.append(initial_date)
                    elif "airleak" in root:
                        airleak_data.append(data)
                        airleak_files.append(file)
                        airleak_dates.append(initial_date) 
    return (airleak_data, no_complication_data), (airleak_files, no_complication_files), (airleak_dates, no_complication_dates)

def pad_data(data, max_rows):
    padded_data = []
    for item in data:
        item = np.pad(item, ((0, max_rows - item.shape[0]), (0, 0)), 'constant')
        padded_data.append(item)
    return np.array(padded_data)

def preprocess_data(airleak_data, no_complication_data, airleak_files, no_complication_files):
    scaler = MinMaxScaler()
    X, y, files = [], [], []

    max_rows = max([item.shape[0] for item in airleak_data + no_complication_data])

    airleak_data = pad_data(airleak_data, max_rows)
    no_complication_data = pad_data(no_complication_data, max_rows)

    for data, file in zip(no_complication_data, no_complication_files):
        data = scaler.fit_transform(data)
        X.append(data)
        y.append(0)
        files.append(file)
    for data, file in zip(airleak_data, airleak_files):
        data = scaler.fit_transform(data)
        X.append(data)
        y.append(1)
        files.append(file)

    return np.array(X), np.array(y), files

def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def find_optimal_threshold(fpr, tpr, thresholds):
    """
    Find the optimal threshold by calculating the point on the ROC curve that has the minimal distance to (0, 1).
    :param fpr: Array of false positive rates.
    :param tpr: Array of true positive rates.
    :param thresholds: Array of corresponding thresholds.
    :return: Optimal threshold.
    """
    distance = ((1 - tpr) ** 2 + fpr ** 2) ** 0.5
    idx = np.argmin(distance)
    return thresholds[idx]

def compute_sensitivity_specificity(y_true, y_pred):
    """
    Compute the sensitivity and specificity given true and predicted labels.
    :param y_true: Array of true labels.
    :param y_pred: Array of predicted labels.
    :return: Tuple (sensitivity, specificity).
    """
    print(f"y_true shape in compute_sensitivity_specificity: {y_true.shape}")  # Add this line
    print(f"y_pred shape in compute_sensitivity_specificity: {y_pred.shape}")  # Add this line
    
    y_pred = y_pred.flatten()  # Add this line
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity

def plot_roc_curve(y_true, y_pred, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f"{output_dir}/roc_curve.png")
    plt.show()

class ThresholdCallback(Callback):
    def __init__(self, X_val, y_val):
        super(ThresholdCallback, self).__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.threshold = 0.5

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val)
        fpr, tpr, thresholds = roc_curve(self.y_val, y_pred)
        self.threshold = find_optimal_threshold(fpr, tpr, thresholds)
        print(f"Optimal Threshold at end of epoch {epoch}: {self.threshold}")  # Print the optimal threshold at end of each epoch

def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def write_classification_results_to_csv(filenames, labels, predictions, prediction_probs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/classification_results.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'classification', 'prediction', 'classification_probability', 'result'])
        for filename, label, prediction, prediction_prob in zip(filenames, labels, predictions, prediction_probs):
            # 以下の行でファイル名の前処理を行う
            filename = filename.replace('til24h-', '')
            writer.writerow([filename, label, prediction, prediction_prob, label == prediction])

def write_row_importance_to_csv(filenames, dates, importances, probabilities, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename, date, importance_map, prediction_prob, label in zip(filenames, dates, importances, probabilities, labels):
        with open(f"{output_dir}/{filename}_importance.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['date', 'importance', 'prediction_prob', 'label'])
            for i, imp in enumerate(importance_map):
                writer.writerow([date, imp, prediction_prob, label])

def compute_grad_cam_importance(model, X_batch, layer_name='conv1d_1'):
    grad_cam_batch = []
    grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])  
    for X in X_batch:
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(np.expand_dims(X, axis=0))
            loss = predictions[0][0]
        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]
        weights = np.mean(grads, axis=0)
        cam = np.dot(output, weights)

        # CAMを入力データの行数に一致するようにアップサンプリング
        scale_factor = len(X) / len(cam)
        cam = zoom(cam, scale_factor)

        cam = (cam - cam.min()) / (cam.max() - cam.min())
        grad_cam_batch.append(cam)
    return np.array(grad_cam_batch)

def save_dataset_info_to_csv(filepaths, labels, split_info, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/dataset_info.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'label', 'split'])
        for filename, label, split in zip(filepaths, labels, split_info):
            writer.writerow([filename, label, split])

def run_classification():
    set_seeds(42)
    (airleak_data, no_complication_data), (airleak_files, no_complication_files), (airleak_dates, no_complication_dates) = load_dataset()
    X, y, filepaths = preprocess_data(airleak_data, no_complication_data, airleak_files, no_complication_files)

    dates= airleak_dates + no_complication_dates
    X_train_val, X_test, y_train_val, y_test, filepaths_train_val, filepaths_test, dates_train_val, dates_test = train_test_split(
        X, y, filepaths, dates, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val, filepaths_train, filepaths_val, dates_train, dates_val = train_test_split(
        X_train_val, y_train_val, filepaths_train_val, dates_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )

    # convert filepaths to filenames
    file_names_test = [Path(fp).name for fp in filepaths_test]

    X_train = np.stack(X_train)
    X_val = np.stack(X_val)
    X_test = np.stack(X_test)

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_cnn_model(input_shape)
    threshold_callback = ThresholdCallback(X_val, y_val)
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), shuffle=True, callbacks=[threshold_callback])
    print(f"Final Optimal Threshold: {threshold_callback.threshold}")  # Print the final optimal threshold after training

    # 重みを保存する
    model.save_weights('model_weights.h5')

    # 手順 6: モデルを評価
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > threshold_callback.threshold).astype(int).flatten()
    
    # Write classification results to csv
    y_pred_probs = y_pred.flatten()  # probabilities
    write_classification_results_to_csv(file_names_test, y_test, y_pred_binary, y_pred_probs, 'output')

    # 感度と特異度を計算
    sensitivity, specificity = compute_sensitivity_specificity(y_test, y_pred_binary.reshape(-1, 1))
    print(f"Sensitivity: {sensitivity}, Specificity: {specificity}")
 
    # Use the function to compute the Grad-CAM importances
    grad_cam_importances = compute_grad_cam_importance(model, X_test)
    
    # Write the importances to CSV
    for filename, date, importance_map, y_pred_single, y_true_single in zip(file_names_test, dates_test, grad_cam_importances, y_pred, y_test):
        write_row_importance_to_csv([filename], [date], [importance_map], [y_pred_single], [y_true_single], 'output')

    # ROC曲線を描画し、AUCを表示
    plot_roc_curve(y_test, y_pred)

    # 手順 7: モデルの性能を可視化
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    run_classification()
