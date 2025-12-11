import os
import re
import numpy as np
import pandas as pd
import torch
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt


def load_data(data_path, label_path, data_len):
    # Read data and label files

    datasets = np.array([])
    labels = np.array([])

    file_list = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.txt')]

    label_list = [os.path.join(label_path, lbl) for lbl in os.listdir(label_path) if lbl.endswith('.txt')]

    for fn in file_list:
        dataset = np.loadtxt(fn)

        # 20250731 Read using Python
        # dataset = np.loadtxt(fn, dtype=float, delimiter=',')
        # Crop data to x*data_len
        dataset = dataset[:, 0:data_len]
        datasets = np.append(datasets, dataset)
        # np.concatenate((datasets, dataset), axis=0)

    for ln in label_list:
        label = np.loadtxt(ln)
        labels = np.append(labels, label)
        # np.concatenate((labels, label), axis=0)

    datasets = datasets.reshape(-1, data_len)

    # labels_matrix = np.zeros((labels.__len__(), 2), dtype=np.float64)

    # labels_matrix[labels == 0] = [1., 0.]
    # labels_matrix[labels == 1] = [0., 1.]

    return datasets, labels


def load_data_csv(data_path, label_path, data_len):
    # Read data and label files

    datasets = np.array([])
    labels = np.array([])

    file_list = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.txt')]

    label_list = [os.path.join(label_path, lbl) for lbl in os.listdir(label_path) if lbl.endswith('.txt')]

    for fn in file_list:
        df = pd.read_csv(fn)
        dataset = df.values
        # Crop data to x*data_len
        dataset = dataset[:, 0:data_len]
        datasets = np.append(datasets, dataset)
        # np.concatenate((datasets, dataset), axis=0)

    for ln in label_list:
        lf = pd.read_csv(ln)
        label = lf.values

        labels = np.append(labels, label)

    datasets = datasets.reshape(-1, data_len)

    return datasets, labels


def data_select(input_data, input_label):
    # Find indices where label is 1, store in All_1_idx
    All_1_idx = np.where(input_label == 1)[0]
    # Length of All_1_idx
    number_ones = len(All_1_idx)
    # Extract data from input_data where label is 1
    out_ones = input_data[All_1_idx]
    # Find indices where label is 0, ensure same count as label 1
    out_zeros_indices = np.where(input_label == 0)[0]
    np.random.shuffle(out_zeros_indices)
    out_zeros = input_data[out_zeros_indices[:number_ones]]
    # Combine data
    out_data = np.concatenate((out_ones, out_zeros))
    out_label = np.concatenate((input_label[All_1_idx], input_label[out_zeros_indices[:number_ones]]))
    return out_data, out_label


def main():
    # Select folders
    print("Select data folder")
    data_folder_selected = filedialog.askdirectory()
    if not data_folder_selected:
        print("No data folder selected")
        return

    print("Select label folder")
    label_folder_selected = filedialog.askdirectory()
    if not label_folder_selected:
        print("No label folder selected")
        return

    print("Select save location")
    save_folder_selected = filedialog.askdirectory()
    if not save_folder_selected:
        print("No save location selected")
        return

    # All_data, All_labels = load_data(data_folder_selected, label_folder_selected, 600)
    All_data, All_labels = load_data_csv(data_folder_selected, label_folder_selected, 600)
    print(f"All Data size:{All_data.shape}, All Label length:{All_labels.shape}.\n")

    Selected_data, Selected_label = data_select(All_data, All_labels)
    print(f"Selected Data size:{str(Selected_data.shape)}, Selected Label length:{Selected_label.shape}.\n")

    np.savetxt(os.path.join(save_folder_selected, "dataset.txt"), Selected_data, '%.5f')
    np.savetxt(os.path.join(save_folder_selected, "label.txt"), Selected_label, '%d')
    print(f"Dataset organized and saved to {save_folder_selected}")


if __name__ == "__main__":
    main()

