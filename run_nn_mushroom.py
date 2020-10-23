import numpy as np
from data_process import construct_MUSHROOM, get_MUSHROOM_data
from models.neural_net import NeuralNetwork, SGD

# TRAINING = 0.6 indicates 60% of the data is used as the training dataset.
VALIDATION = 0.2

construct_MUSHROOM()
data = get_MUSHROOM_data(VALIDATION)
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']
n_class = len(np.unique(y_test))

print("Number of train samples: ", X_train.shape[0])
print("Number of val samples: ", X_val.shape[0])
print("Number of test samples: ", X_test.shape[0])


# ### Get Accuracy

# This function computes how well your model performs using accuracy as a metric.
def train(net, epochs, batch_size, learning_rate, learning_rate_decay, regularization):
    # Variables to store performance for each epoch
    train_loss = np.zeros(epochs)
    train_accuracy = np.zeros(epochs)
    val_accuracy = np.zeros(epochs)

    # For each epoch...
    for epoch in range(epochs):
        print('epoch:', epoch)

        # Shuffle the dataset

        # Training
        train_accuracy[epoch] += net.forward_accuracy(X_train, y_train)
        # For each mini-batch...
        for batch in range(X_train.shape[0] // batch_size):
            # Create a mini-batch of training data and labels
            X_batch = X_train[batch*batch_size: (batch+1) * batch_size]
            y_batch = y_train[batch*batch_size: (batch+1) * batch_size]

            # Run the forward pass of the model to get a prediction and compute the accuracy
            # Run the backward pass of the model to update the weights and compute the loss
            train_loss[epoch] += net.backward(X_batch, y_batch, learning_rate * learning_rate_decay ** epoch, reg=regularization)

        # Validation
        # No need to run the backward pass here, just run the forward pass to compute accuracy
        val_accuracy[epoch] += net.forward_accuracy(X_val, y_val)

        print('train_loss:', train_loss[epoch])
        print('train_accuracy:', train_accuracy[epoch])
        print('val_accuracy:', val_accuracy[epoch])

    return train_loss, train_accuracy, val_accuracy


# Hyperparameters
input_size = X_train.shape[1]
num_layers = 3
hidden_size = 128
hidden_sizes = [hidden_size] * (num_layers - 1)
num_classes = n_class
epochs = 100
batch_size = 200
learning_rate = 0.1
learning_rate_decay = 0.99
regularization = 0.01

# Initialize a new neural network model
sgd_2_layer = NeuralNetwork(input_size, hidden_sizes, num_classes, num_layers, optimizer=SGD, norm_weights=True)

sgd_2_layer_train_loss, sgd_2_layer_train_accuracy, sgd_2_layer_val_accuracy = train(sgd_2_layer, epochs, batch_size, learning_rate, learning_rate_decay, regularization)
