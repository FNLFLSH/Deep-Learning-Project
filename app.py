from flask import Flask, render_template, request, jsonify
import threading
import numpy as np
import torchvision
import torchvision.transforms as transforms
from CNN_team_number import CNN
from MLP_team_number import MLP

# Initialize Flask app
app = Flask(__name__)

# Global variables to store model results
cnn_result = None
mlp_result = None
cnn_running = False
mlp_running = False

def train_cnn():
    global cnn_result, cnn_running
    cnn_running = True
    
    # Import necessary components from Working_Cnn.py
    from Working_Cnn import load_data
    
    train_loader, test_loader = load_data()
    
    input_size = 28
    num_epochs = 5
    model = CNN(input_size=28, num_filters=1, kernel_size=3, fc_output_size=10, lr=0.01)
    
    # Training
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            x = inputs.squeeze(1).numpy()
            y = labels.numpy()
            loss = model.train(x, y)
            total_loss += loss
    
    # Testing
    correct_pred = 0
    total_pred = 0
    for inputs, labels in test_loader:
        x = inputs.squeeze(1).numpy()
        y = labels.numpy()
        pred = model.forward(x)
        predicted_labels = np.argmax(pred, axis=1)
        correct_pred += np.sum(predicted_labels == y)
        total_pred += len(labels)
    
    accuracy = float(correct_pred/total_pred)
    cnn_result = {
        "accuracy": accuracy,
        "epochs": num_epochs,
        "model_type": "CNN"
    }
    cnn_running = False

def train_mlp():
    global mlp_result, mlp_running
    mlp_running = True
    
    # Import necessary components from MLP.py
    from MLP import load_data
    
    train_loader, test_loader = load_data()
    
    input_size = 28 * 28
    hidden_size = 128
    output_size = 10
    learning_rate = 0.01
    num_epochs = 10
    
    model = MLP(input_size, hidden_size, output_size, learning_rate)
    
    # Training
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            x_batch = inputs.view(-1, input_size).numpy()
            y_batch = labels.numpy()
            loss = model.train(x_batch, y_batch)
            total_loss += loss
    
    # Testing
    correct_pred = 0
    total_pred = 0
    for inputs, labels in test_loader:
        x = inputs.view(-1, input_size).numpy()
        y = labels.numpy()
        pred = model.forward(x)
        predicted_labels = np.argmax(pred, 1)
        correct_pred += np.sum(predicted_labels == y)
        total_pred += len(labels)
    
    accuracy = float(correct_pred/total_pred)
    mlp_result = {
        "accuracy": accuracy,
        "epochs": num_epochs,
        "model_type": "MLP"
    }
    mlp_running = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cnn')
def cnn_page():
    return render_template('cnn.html')

@app.route('/mlp')
def mlp_page():
    return render_template('mlp.html')

@app.route('/run_cnn', methods=['POST'])
def run_cnn():
    global cnn_result, cnn_running
    
    if cnn_running:
        return jsonify({"status": "running"})
    
    if cnn_result is None:
        # Start the CNN training in a separate thread
        threading.Thread(target=train_cnn).start()
        return jsonify({"status": "started"})
    else:
        return jsonify({"status": "complete", "result": cnn_result})

@app.route('/run_mlp', methods=['POST'])
def run_mlp():
    global mlp_result, mlp_running
    
    if mlp_running:
        return jsonify({"status": "running"})
    
    if mlp_result is None:
        # Start the MLP training in a separate thread
        threading.Thread(target=train_mlp).start()
        return jsonify({"status": "started"})
    else:
        return jsonify({"status": "complete", "result": mlp_result})

@app.route('/check_cnn_status')
def check_cnn_status():
    global cnn_result, cnn_running
    
    if cnn_running:
        return jsonify({"status": "running"})
    elif cnn_result is not None:
        return jsonify({"status": "complete", "result": cnn_result})
    else:
        return jsonify({"status": "not_started"})

@app.route('/check_mlp_status')
def check_mlp_status():
    global mlp_result, mlp_running
    
    if mlp_running:
        return jsonify({"status": "running"})
    elif mlp_result is not None:
        return jsonify({"status": "complete", "result": mlp_result})
    else:
        return jsonify({"status": "not_started"})

if __name__ == '__main__':
    app.run(debug=True) 