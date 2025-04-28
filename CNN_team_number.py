import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ===================== Utility Functions ===================== #

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def convolve2d(img, kernel):
    kernel_height, kernel_width = kernel.shape
    img_height, img_width = img.shape
    output_height = img_height - kernel_height + 1
    output_width = img_width - kernel_width + 1
    output = np.zeros((output_height, output_width))
    for i in range(output_height):
        for j in range(output_width):
            region = img[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = np.sum(region * kernel)
    return output

# ===================== Data Loading ===================== #

def dataloader(train_dataset, test_dataset, batch_size=128):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transform)
    print("Training samples:", len(train_dataset))
    print("Testing samples:", len(test_dataset))
    return dataloader(train_dataset, test_dataset)

# ===================== CNN Structure ===================== #

class CNN:
    def __init__(self, input_size, num_filters, kernel_size, fc_output_size, lr):
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.lr = lr

        self.conv_kernel = np.random.randn(num_filters, kernel_size, kernel_size) * 0.1

        conv_output_dim = input_size - kernel_size + 1
        flattened_size = num_filters * conv_output_dim * conv_output_dim

        self.fc_weights = np.random.randn(flattened_size, fc_output_size) * 0.1
        self.fc_bias = np.zeros((1, fc_output_size))

    def forward(self, x):
        self.batch_size = x.shape[0]
        self.input = x

        self.conv_out = np.zeros((self.batch_size, self.num_filters, x.shape[1] - self.kernel_size + 1, x.shape[2] - self.kernel_size + 1))
        for i in range(self.batch_size):
            for f in range(self.num_filters):
                self.conv_out[i, f] = convolve2d(x[i], self.conv_kernel[f])

        self.relu_out = relu(self.conv_out)
        self.flatten = self.relu_out.reshape(self.batch_size, -1)
        logits = np.dot(self.flatten, self.fc_weights) + self.fc_bias
        self.probs = softmax(logits)
        return self.probs

    def backward(self, x, y, pred):
        y_onehot = np.zeros_like(pred)
        y_onehot[np.arange(self.batch_size), y] = 1

        d_logits = (pred - y_onehot)

        dW_fc = np.dot(self.flatten.T, d_logits)
        db_fc = np.sum(d_logits, axis=0, keepdims=True)
        d_flatten = np.dot(d_logits, self.fc_weights.T)
        d_relu_out = d_flatten.reshape(self.relu_out.shape)
        d_conv_out = d_relu_out * (self.conv_out > 0)

        d_kernel = np.zeros_like(self.conv_kernel)
        for f in range(self.num_filters):
            for i in range(self.batch_size):
                d_kernel[f] += convolve2d(x[i], d_conv_out[i, f])

        d_kernel /= self.batch_size
        dW_fc /= self.batch_size
        db_fc /= self.batch_size

        self.fc_weights -= self.lr * dW_fc
        self.fc_bias -= self.lr * db_fc
        self.conv_kernel -= self.lr * d_kernel

    def train(self, x, y):
        preds = self.forward(x)
        loss = -np.sum(np.log(preds[np.arange(len(y)), y] + 1e-9)) / len(y)
        self.backward(x, y, preds)
        return loss

# ===================== Training Process ===================== #

def main():
    train_loader, test_loader = load_data()

    input_size = 28
    num_epochs = 5
    model = CNN(input_size=28, num_filters=1, kernel_size=3, fc_output_size=10, lr=0.01)

    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            x = inputs.squeeze(1).numpy()
            y = labels.numpy()
            loss = model.train(x, y)
            total_loss += loss

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

    correct_pred = 0
    total_pred = 0
    for inputs, labels in test_loader:
        x = inputs.squeeze(1).numpy()
        y = labels.numpy()
        pred = model.forward(x)
        predicted_labels = np.argmax(pred, axis=1)
        correct_pred += np.sum(predicted_labels == y)
        total_pred += len(labels)
    print(f"Test Accuracy: {correct_pred/total_pred:.4f}")

if __name__ == "__main__":
    main()
