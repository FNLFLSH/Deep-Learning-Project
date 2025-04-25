# Deep Learning Project - MNIST Classification

This project implements two neural network models (MLP and CNN) for handwritten digit classification on the MNIST dataset using Python 3.9.

## Requirements

- Python 3.9
- numpy
- torchvision
- torch

## Installation

1. Install Python 3.9
2. Install required packages:
```bash
pip install numpy torchvision torch
```

## Project Structure

- `MLP_team_number.py`: Implementation of Multi-Layer Perceptron
- `CNN_team_number.py`: Implementation of Convolutional Neural Network
- `app.py`: Flask web interface for running models

## Running the Models

### Individual Models

To run the MLP model:
```bash
python MLP_team_number.py
```

To run the CNN model:
```bash
python CNN_team_number.py
```

### Web Interface

To run the web interface:
```bash
python app.py
```
Then open your browser and navigate to `http://127.0.0.1:5000/`

## Model Specifications

### MLP Model
- 2-layer neural network
- First layer: Fully connected with sigmoid activation
- Second layer: Fully connected with 10 neurons and softmax activation
- Batch size: 128
- Maximum epochs: 100

### CNN Model
- 2-layer neural network
- First layer: Convolutional layer with ReLU activation
- Second layer: Fully connected with 10 neurons and softmax activation
- Batch size: 128
- Maximum epochs: 5

## Implementation Details

Both models are implemented from scratch without using any deep learning APIs (torch.nn, tf.keras, etc.). The implementations include:
- Forward propagation
- Backward propagation with gradient calculations
- Training functions
- Activation functions (Sigmoid/ReLU and Softmax)
- Cross-entropy loss
- Main training and testing loops