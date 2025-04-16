import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def sigmoid(x):
    """
    Sigmoid activation function.

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array after applying sigmoid.
    """
    return 1 / (1 + np.exp(-x))

def softmax(x):
    """
    Softmax activation function.

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array after applying softmax (probabilities).
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def dataloader(train_dataset, test_dataset, batch_size=128):
    """
    Creates DataLoader instances for training and testing datasets.

    Args:
        train_dataset (torch.utils.data.Dataset): Training dataset.
        test_dataset (torch.utils.data.Dataset): Testing dataset.
        batch_size (int): Size of each batch.

    Returns:
        Tuple[DataLoader, DataLoader]: Tuple containing the training and testing DataLoaders.
    """
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def load_data():
    """
    Loads the MNIST dataset and creates DataLoaders.

    Returns:
        Tuple[DataLoader, DataLoader]: Tuple containing the training and testing DataLoaders.
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) # Using the normalization from your working code
    train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transform)
    print("The number of training data:", len(train_dataset))
    print("The number of testing data:", len(test_dataset))
    return dataloader(train_dataset, test_dataset)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        """
        Initializes the Multi-layer Perceptron.

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of neurons in the hidden layer.
            output_size (int): The number of neurons in the output layer.
            lr (float): The learning rate for gradient descent.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = lr

        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, x):
        """
        Forward propagation function.

        Args:
            x (np.ndarray): Input batch of flattened MNIST images (shape [batch_size, input_size]).

        Returns:
            np.ndarray: Output probabilities after softmax.
        """
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        outputs = softmax(self.z2)
        return outputs

    def backward(self, x, y_true, y_pred):
        """
        Backward propagation function (including gradient calculations and parameter updates).

        Args:
            x (np.ndarray): Input batch of images.
            y_true (np.ndarray): Ground-truth labels (one-hot encoded).
            y_pred (np.ndarray): Prediction probabilities from forward propagation.
        """
        m = x.shape[0]

        # Gradient of the loss with respect to the output of the second layer (z2)
        dz2 = y_pred - y_true

        # Gradient of the loss with respect to the weights and biases of the second layer
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Gradient of the loss with respect to the output of the first layer (a1)
        da1 = np.dot(dz2, self.W2.T)

        # Gradient of the loss with respect to the input of the first layer (z1)
        dz1 = da1 * self.a1 * (1 - self.a1)  # Derivative of sigmoid

        # Gradient of the loss with respect to the weights and biases of the first layer
        dW1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update the weights and biases
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def cross_entropy_loss(self, y_pred, y_true):
        """
        Calculates the cross-entropy loss.

        Args:
            y_pred (np.ndarray): Predicted probabilities.
            y_true (np.ndarray): True labels (one-hot encoded).

        Returns:
            float: The cross-entropy loss.
        """
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return loss

    def train(self, x, y_true):
        """
        Trains the MLP on a single batch of data.

        Args:
            x (np.ndarray): Input batch.
            y_true (np.ndarray): True labels (not one-hot encoded).

        Returns:
            float: The loss for the current batch.
        """
        # Forward propagation
        y_pred = self.forward(x)

        # One-hot encode the labels
        y_one_hot = np.zeros((y_true.shape[0], self.output_size))
        y_one_hot[np.arange(y_true.shape[0]), y_true] = 1

        # Calculate loss
        loss = self.cross_entropy_loss(y_pred, y_one_hot)

        # Backward propagation
        self.backward(x, y_one_hot, y_pred)

        return loss

def main():
    # First, load data
    train_loader, test_loader = load_data()

    # Second, define hyperparameters
    input_size = 28 * 28  # MNIST images are 28x28 pixels
    hidden_size = 128
    output_size = 10
    learning_rate = 0.01
    num_epochs = 10

    # Then, initialize the model
    model = MLP(input_size, hidden_size, output_size, learning_rate)

    # Train the model
    print("Starting training...")
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            # Flatten the input images
            x_batch = inputs.view(-1, input_size).numpy()
            y_batch = labels.numpy()

            # Train on the batch
            loss = model.train(x_batch, y_batch)
            total_loss += loss

            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Step [{i+1}/{len(train_loader)}], Loss: {loss:.4f}")

        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss/len(train_loader):.4f}") # print the loss for each epoch

    # Finally, evaluate the model
    correct_pred = 0
    total_pred = 0
    for inputs, labels in test_loader:
        x = inputs.view(-1, input_size).numpy()
        y = labels.numpy()
        pred = model.forward(x)  # the model refers to the model that was trained during the raining phase
        predicted_labels = np.argmax(pred, 1)
        correct_pred += np.sum(predicted_labels == y)
        total_pred += len(labels)
    print(f"Test Accuracy: {correct_pred/total_pred:.4f}")

if __name__ == "__main__":  # Program entry
    main()
