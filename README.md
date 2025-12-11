# ⚡ Transient Current Waveform Classification

> A transient current waveform classification tool using 1D-CNN

## 📋 Project Overview

This project uses **1D Convolutional Neural Networks (1D-CNN)** for automatic classification of transient current waveforms.

## 🗂️ Project Structure

```
TransientCNN/
├── code/                          # Core code directory
│   ├── mainCNN.py                # Main training program 🎯
│   ├── best_model_CNN.pth        # Pre-trained model weights 💾
│   ├── data_labeling_gui_english.py   # Data labeling tool 🏷️
│   ├── Making_Datasets.py        # Dataset preparation script 📊
│   ├── model_processing.py       # Model processing utilities 🔧
│   ├── evalue_model_noise_resisitance.py   # Noise resistance evaluation 🔊
│   └── evalue_model_time_consuming.py      # Performance time evaluation ⏱️
└── dataset/                       # Dataset directory
    ├── dataset/                   # Training data 
    └── label/                     # Label data
```

## ✨ Main Features

- 🧠 **Binary Classification**: 1D-CNN based transient current waveform classification
- 🖥️ **Visual Labeling**: GUI tool for data annotation
- 📈 **Model Evaluation**: Noise resistance and runtime performance evaluation
- 🎓 **End-to-End Pipeline**: Complete workflow from data preprocessing to model training

## 🚀 Quick Start

### Train Model

```bash
cd code
python mainCNN.py
```

### Label Data

```bash
python data_labeling_gui_english.py
```

## 📊 Dataset Notes

⚠️ **Note**: The dataset in the `dataset/` directory is for model training demonstration only. Additional test datasets are being organized and will be uploaded soon.

## 🛠️ Tech Stack

- **Deep Learning Framework**: PyTorch
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib
- **GUI Development**: Tkinter

## 📝 License

This project is for academic and research purposes only.

---

⭐ If this project helps you, feel free to star it!
