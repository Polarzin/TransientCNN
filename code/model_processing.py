import os
import re

import numpy as np
import pandas as pd
import torch
import tkinter as tk
from tkinter import filedialog

from matplotlib import pyplot as plt

from mainCNN import WaveformCNN


# Load model
def load_model(model_path):
    model = torch.load(model_path, weights_only=False)
    model.eval()  # Set model to evaluation mode
    return model


# Preprocess input data (modify based on needs)
def preprocess_input(input_data, data_len):
    #  (batch, channel, length); batch size, channel count, data length
    data_num = input_data.shape
    if len(data_num) > 1:
        input_data = input_data[:, 0:data_len]
    else:
        input_data = input_data[0:data_len]
    return input_data


def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    ree = np.zeros(interval.shape)
    data_num = interval.shape
    if len(data_num) > 1:
        for i in range(interval.shape[0]):
            # Process row-wise
            ree[i, :] = np.convolve(interval[i, :], window, 'same')
    else:
        if interval.size == 0:
            return np.array([])
        else:
            ree = np.convolve(interval, window, 'same')
    return ree


def normalization(data):
    if data.size != 0:
        data_num = data.shape
        if len(data_num) > 1:
            data_max_val = data.max(axis=1)
            data_min_val = data.min(axis=1)
            # print(data_max_val.shape)
            data_max_val = data_max_val.reshape([data_max_val.shape[0], 1])
            data_min_val = data_min_val.reshape([data_min_val.shape[0], 1])
            # print(data_max_val.shape)
        else:
            data_max_val = data.max()
            data_min_val = data.min()
        data_norm = (data - data_min_val) / (data_max_val - data_min_val)  # Column 0, row 1. Analyze based on specific case
    else:
        data_norm = np.array([])
    return data_norm


def sample_array(array, step):
    """
    Sample 2D array.

    Args:
    array (numpy.ndarray): Input 2D array.
    step (int): Sampling step.

    Returns:
    numpy.ndarray: Sampled 2D array.
    """
    if step == 0:
        return array
    else:
        data_num = array.shape
        if len(data_num) > 1:
            return array[:, ::step]
        else:
            return array[::step]


# Post-process model output (modify based on needs)
def postprocess_output(output):
    train_predicted = torch.max(output.data, 1).indices
    Label_1_Num = (train_predicted == 1).sum().item()
    train_predicted = train_predicted.to("cpu")
    train_predicted = train_predicted.numpy()

    # Convert output to string
    return train_predicted, Label_1_Num


# Main function
def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sequence length
    sequence_length = 300

    # Load model
    # model_path = 'Pristine-best_model_condition_CNN-Ac92-V2.pth'
    model_path = 'Best_BABr_SXY-best_model_CNN.pth'
    model = load_model(model_path)

    # Create Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide main window

    # Select folder
    folder_selected = filedialog.askdirectory()
    if not folder_selected:
        print("No folder selected")
        return

    result_array = []

    # Iterate all txt files in folder
    for filename in os.listdir(folder_selected):
        if filename.endswith('.txt'):

            print(f"File Processing: {filename}")

            channel_number = re.findall(r'\d+', filename)[-1]
            channel_number = int(channel_number)

            file_path = os.path.join(folder_selected, filename)

            # For tcData saved as CSV, use the following two lines to read
            if os.path.getsize(file_path) == 0:
                channel_tc_data = np.array([])
            else:
                channel_tc_data = pd.read_csv(file_path, header=None)
                channel_tc_data = channel_tc_data.values

            # Store output results for each input row
            outputs = []
            input_data = preprocess_input(channel_tc_data, 600)
            # input_data = moving_average(input_data, 5)
            input_data = normalization(input_data)

            if input_data.size > 0:
                # input_data = sample_array(input_data, step=int(input_data.shape[1] / sequence_length))

                if len(input_data.shape) > 1:
                    input_data = sample_array(input_data, step=int(input_data.shape[1] / sequence_length))
                else:
                    input_data = sample_array(input_data, step=int(len(input_data) / sequence_length))

                # Plot input for debug
                # plot_batch_data = input_data
                # plt.plot(plot_batch_data.T)
                # plt.show(block=True)

                input_data = torch.tensor(input_data, dtype=torch.float32).view(-1, 1, sequence_length)  # Add batch dimension

                # for line_data in range(input_data.shape[0]):
                with torch.no_grad():
                    output = model(input_data.to(device))
                output_predicted_np, label1_number = postprocess_output(output)

                # plot_batch_data = input_data.to("cpu")
                # plot_batch_data = plot_batch_data.view(-1, sequence_length).numpy()
                # plt.plot(plot_batch_data[0, :].T)
                # print(label1_number(0))
                # plt.show(block=True)
            else:
                output_predicted_np = np.array(0)
                label1_number = np.array(0)

            outputs.append(output_predicted_np)
            result_array.append([channel_number, label1_number])

            # Save output to new txt file
            output_filename = f"{os.path.splitext(filename)[0]}_output.txt"
            output_path = os.path.join(folder_selected, "details", output_filename)
            np.savetxt(output_path, np.array(outputs), fmt="%d")

    result_filename = "result.txt"
    # result_path = os.path.join(folder_selected, "output", result_filename)
    result_path = os.path.join(folder_selected, result_filename)
    np.savetxt(result_path, np.array(result_array), fmt="%d")
    print("Processing completed")


if __name__ == "__main__":
    main()
