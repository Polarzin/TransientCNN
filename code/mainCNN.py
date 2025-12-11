import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


class WaveformCNN(nn.Module):
    def __init__(self, input_length=600, num_classes=2):
        super(WaveformCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(64 * (input_length // 4), 128)  # After two pooling ops, length halved twice
        self.fc2 = nn.Linear(128, num_classes)  # For binary classification, num_classes=2

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(-1, 64 * (x.size(2)))  # Flatten

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        # For binary classification, can use sigmoid with threshold, but usually handled in loss function
        # x = torch.sigmoid(x)
        x = nn.functional.softmax(x, dim=1)
        return x


class WaveformDataset(Dataset):
    def __init__(self, data, labels):
        """
        Args:
            data (numpy.ndarray): A 2D numpy array of shape (20000, sequence_length) containing the raw data.
            labels (numpy.ndarray): A 2D numpy array of shape (20000, 1) containing the labels.
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, :], self.labels[idx]  # Return the data and the label (squeeze the label dimension)


def load_data(data_path, label_path, data_len):
    # Read data and label files

    datasets = np.array([])
    labels = np.array([])

    file_list = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.txt')]

    label_list = [os.path.join(label_path, lbl) for lbl in os.listdir(label_path) if lbl.endswith('.txt')]

    for fn in file_list:
        dataset = np.loadtxt(fn)
        # Crop data to x*data_len
        dataset = dataset[:, 0:data_len]
        datasets = np.append(datasets, dataset)
        # np.concatenate((datasets, dataset), axis=0)

    for ln in label_list:
        label = np.loadtxt(ln)
        labels = np.append(labels, label)
        # np.concatenate((labels, label), axis=0)

    datasets = datasets.reshape(-1, data_len)

    labels_matrix = np.zeros((labels.__len__(), 2), dtype=np.float64)

    labels_matrix[labels == 0] = [1., 0.]
    labels_matrix[labels == 1] = [0., 1.]

    return datasets, labels_matrix


def normalization(data):
    data_max_val = data.max(axis=1)
    data_min_val = data.min(axis=1)
    # print(data_max_val.shape)
    data_max_val = data_max_val.reshape([data_max_val.shape[0], 1])
    data_min_val = data_min_val.reshape([data_min_val.shape[0], 1])
    # print(data_max_val.shape)
    data_norm = (data - data_min_val) / (data_max_val - data_min_val)  # axis=1 for row-wise, analyze based on specific case
    return data_norm


def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    re = np.zeros(interval.shape)
    data_num = interval.shape
    if len(data_num) > 1:
        for i in range(interval.shape[0]):
            # Process row-wise
            re[i, :] = np.convolve(interval[i, :], window, 'same')
    else:
        re = np.convolve(interval, window, 'same')
    return re


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
        return array[:, ::step]


def data_select(input_data, input_label):
    # Find indices where label is 1, store in All_1_idx
    All_1_idx = np.where(input_label == 1)[0]
    # Length of All_1_idx
    number_ones = len(All_1_idx)
    # Extract data from input_data where label is 1
    out_ones = input_data[All_1_idx]
    # Find indices where label is 0, randomly select same number as label 1
    out_zeros_indices = np.where(input_label == 0)[0]
    np.random.shuffle(out_zeros_indices)
    out_zeros = input_data[out_zeros_indices[:number_ones]]
    # Combine into output
    out_data = np.concatenate((out_ones, out_zeros))
    out_label = np.concatenate((input_label[All_1_idx], input_label[out_zeros_indices[:number_ones]]))
    return out_data, out_label


if __name__ == "__main__":
    # Hyperparameters
    num_classes = 2  # Number of classes
    sequence_length = 300  # Sequence length
    my_batch_size = 64  # Batch size
    learning_rate = 0.00005  # Learning rate
    num_epochs = 1000  # Training epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = "../dataset/dataset/"
    label_path = "../dataset/label/"

    All_data, All_labels = load_data(data_path, label_path, 600)

    print(All_data.shape)

    # Smooth raw data
    # All_data = moving_average(All_data, 5)

    # Sampling
    All_data = sample_array(All_data, step=int(All_data.shape[1] / sequence_length))

    # Define the sizes for the training set, validation set, and test set
    train_size = int(0.8 * len(All_data))
    val_size = int(0.1 * len(All_data))
    test_size = len(All_data) - train_size - val_size

    indices = np.random.permutation(len(All_data))
    All_data = All_data[indices]
    All_labels = All_labels[indices]

    # Normalization
    All_data = normalization(All_data)

    # Create the dataset
    dataset = WaveformDataset(All_data, All_labels)

    # Create the datasets using random_split
    train_dataset, val_test_dataset = random_split(dataset, [train_size, val_size + test_size])
    val_size_2, test_size_2 = len(val_test_dataset) // 2, len(val_test_dataset) - len(val_test_dataset) // 2
    val_dataset, test_dataset = random_split(val_test_dataset, [val_size_2, test_size_2])

    # Create DataLoaders for each dataset
    train_loader = DataLoader(train_dataset, batch_size=my_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=my_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=my_batch_size, shuffle=False)

    # Initialize model, loss function and optimizer
    model = WaveformCNN(sequence_length, num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):

        model.train()

        running_loss = 0.0
        train_correct = 0.0
        train_total = 0

        for idx, (batch_data, batch_label) in enumerate(train_loader):  # gives batch data
            # Training
            # If still incorrect, plot all data in batch to check variance and label correspondence
            # Write loop here, can reduce batch size (16/32)
            # Print to verify
            # batch_data here
            '''
            if (epoch == 10) and (idx == 0):
                plot_batch_data = batch_data.to("cpu")
                plot_batch_data = plot_batch_data.numpy()
                plt.plot(plot_batch_data.T)
                plt.show(block=True)
                A=2
            '''

            batch_data = batch_data.view(-1, 1,
                                         sequence_length)  # reshape to (batch, channel, length); batch size, channel count, data length
            batch_label = batch_label.view(-1, 2)  # reshape to (batch, input_size)

            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)

            optimizer.zero_grad()
            # if epoch==15:
            #    outputs = model(batch_data)
            #    a=1
            # else:
            #    outputs = model(batch_data)

            outputs = model(batch_data)

            loss = criterion(outputs, batch_label)
            loss.backward()
            optimizer.step()

            _, train_predicted = torch.max(outputs.data, 1)

            train_real_label = torch.max(batch_label.data, 1).indices

            running_loss += loss.item() * batch_data.size(0)

            # Calculate accuracy
            train_total += batch_label.size(0)
            train_correct += (train_predicted == train_real_label).sum().item()

        # Print info
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {(running_loss / len(train_loader.dataset)):.4f}, '
            f'ACC: {train_correct / train_total * 100:.3f}%')

        val_total = 0
        val_correct = 0

        model.eval()
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:  # or test_loader for testing
                val_inputs = val_inputs.view(-1, 1, sequence_length)  # reshape to (batch, time_step, input_size)

                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_inputs)

                # max_indices = torch.argmax(outputs, dim=1)
                # bs = outputs.size(0)
                # result = torch.zeros(bs, 2, dtype=torch.float32).to(device)
                # result.scatter_(1, max_indices.unsqueeze(1), 1)
                # equal_rows = torch.all(result == val_labels, dim=1)
                # num_equal_rows = torch.sum(equal_rows).item()
                # ratio = num_equal_rows / my_batch_size

                _, val_predicted = torch.max(val_outputs.data, 1)

                val_real_label = torch.max(val_labels.data, 1).indices

                # Calculate accuracy
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_real_label).sum().item()

                val_loss = criterion(val_outputs, val_labels)
                val_loss_cpu = val_loss.to("cpu")

                # accuracy = float((mdl_predi_labels == val_labels).astype(int).sum()) / float(mdl_predi_labels.size)
        print('Epoch: ', epoch + 1, '| val loss: %.4f' % val_loss_cpu.data.numpy(), 'ac_ratio: ',
              val_correct / val_total)

    # Test model
    # Set model to evaluation mode
    model.eval()

    # Initialize validation metrics
    test_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient computation for validation
    with torch.no_grad():
        for test_inputs, test_labels in test_loader:
            test_inputs = test_inputs.view(-1, 1, sequence_length)  # reshape to (batch, time_step, input_size)

            # Move data to model device (e.g., GPU)
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

            # Forward pass
            test_outputs = model(test_inputs)

            # Calculate loss (not always needed in validation, but useful for monitoring)
            test_loss = criterion(test_outputs, test_labels)
            test_loss += test_loss.item() * test_inputs.size(0)  # Accumulate batch loss

            # Get predictions (for classification, typically argmax)
            # This section goes up
            _, test_predicted = torch.max(test_outputs.data, 1)

            real_label = torch.max(test_labels.data, 1).indices

            # Calculate accuracy
            total += test_labels.size(0)
            correct += (test_predicted == real_label).sum().item()

    # Calculate average loss and accuracy
    average_test_loss = test_loss / test_loader.__len__()
    test_accuracy = 100 * correct / total

    print(f'Validation Loss: {average_test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')

    torch.save(model, 'best_model_CNN.pth')
