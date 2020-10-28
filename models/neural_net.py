"""Neural network model."""

from collections import Counter
from functools import reduce
from typing import Sequence, Callable, Dict, List, Type

import numpy as np
np.random.seed(0)


"""
================= Activation Functions =========================
"""


class ActivationFunction:
    @classmethod
    def __call__(cls, activation_input: np.ndarray) -> np.ndarray:
        return cls.evaluate(activation_input)

    @classmethod
    def evaluate(cls, activation_input: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def evaluate_grad(cls, grad_output: np.ndarray, activated_out) -> np.ndarray:
        pass


class ReLU(ActivationFunction):
    """Computes the ReLU function on the input
    activation_input: 1D array with shape (size,)
    returns: 1D array with same shape as input
    """

    @classmethod
    def evaluate(cls, activation_input: np.ndarray) -> np.ndarray:
        return np.maximum(activation_input, 0)

    @classmethod
    def evaluate_grad(cls, grad_output: np.ndarray, activated_out: np.ndarray) -> np.ndarray:
        new_grad = grad_output.copy()
        new_grad[activated_out == 0] = 0
        return new_grad


class Softmax(ActivationFunction):
    """Computes the softmax function on the input
    Z: (N,d) array
    returns: array with same shape as input
    """

    @classmethod
    def evaluate(cls, Z: np.ndarray) -> np.ndarray:
        shifted = Z - np.max(Z, axis=1).reshape(len(Z), 1)
        amplitude = np.exp(shifted)
        norm = np.sum(amplitude, axis=1).reshape(len(Z), 1)

        return amplitude / norm

    @classmethod
    def evaluate_grad(cls, grad_output: np.ndarray, activated_out: np.ndarray) -> np.ndarray:
        raise Exception("Unimplemented")  # Not needed: cancels with cross entropy


class Identity(ActivationFunction):
    """Computes the identity map.
    activation_input: 1D array with shape (size,)
    returns: 1D array with same shape as input
    """

    @classmethod
    def evaluate(cls, activation_input: np.ndarray) -> np.ndarray:
        return activation_input

    @classmethod
    def evaluate_grad(cls, grad_output: np.ndarray, activated_out: np.ndarray) -> np.ndarray:
        return np.full(grad_output.shape, 1)


"""
================= Loss Functions =========================
"""


class LossFunction:
    @classmethod
    def __call__(cls, y: np.ndarray, y_pred: np.ndarray, reg: float, weights) -> float:
        return cls.evaluate(y, y_pred, reg, weights)

    @classmethod
    def evaluate(cls, y: np.ndarray, y_pred: np.ndarray, reg: float, weights) -> float:
        raise Exception("UNIMPLEMENTED")

    @classmethod
    def evaluate_grad(cls, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise Exception("UNIMPLEMENTED")


class CrossEntropy(LossFunction):
    @classmethod
    def evaluate(cls, y: np.ndarray, y_pred: np.ndarray, reg: float, weights) -> float:
        """Compute the Cross Entropy Loss by comparing the Softmax outputs to the one hot encoded labels

        Parameters:
            y: The softmax outputs of shape (N, C)
            y_pred: The one hot labels of shape (N, C)

        Returns:
            The Cross Entropy Loss
        """
        reg_cost = sum(np.linalg.norm(w) for w in weights) * (reg / (2 * y.shape[0]))

        return - np.sum(y * np.ma.log(y_pred).filled(0)) + reg_cost

    @classmethod
    def evaluate_grad_with_softmax(cls, y: np.ndarray, scores: np.ndarray) -> np.ndarray:
        sum_of_scores = np.sum(scores, axis=1)
        correct_scores = scores[range(len(y)), y]

        d_layer = scores
        d_layer[range(len(y)), y] = (correct_scores - sum_of_scores) / sum_of_scores
        d_layer /= len(y)
        return d_layer


"""
================= NeuralNet Layers =========================
"""


class Layer:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.linear_in = np.zeros(input_size)
        self.linear_out = np.zeros(input_size)
        self.activated_out = np.zeros(output_size)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        raise Exception("UNIMPLEMENTED")

    def update(self, grad: np.ndarray = None) -> None:
        raise Exception("UNIMPLEMENTED")


class LinearLayer(Layer):
    def __init__(self, input_size: int, output_size: int, activation_func: Callable = Identity):
        super().__init__(input_size, output_size)

        self.activation_func = activation_func

        self.w = 1e-4 * np.random.randn(input_size, output_size)
        self.b = np.zeros(output_size)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Computes the softmax function on the input
        x: 1D array with shape (size,)
        returns: 1D array with same shape as input
        """
        # Store input and output data for use in backward pass
        self.linear_in = x
        self.linear_out = x.dot(self.w) + self.b
        self.activated_out = self.activation_func.evaluate(self.linear_out)

        return self.activated_out

    def update(self, w_grad: np.ndarray, b_grad: np.ndarray, lr: float) -> None:
        self.w -= lr * w_grad
        self.b -= lr * b_grad


"""
================= Model optimizers =========================
"""


class Optimizer:
    def update(self, weight_grads: List[np.ndarray], bias_grads: List[np.ndarray], lr: float) -> None:
        raise Exception("Unimplemented")


class SGD(Optimizer):
    def update(self, layers, weight_grads: List[np.ndarray], bias_grads: List[np.ndarray], lr: float) -> None:
        for layer, dW, db in zip(layers, weight_grads, bias_grads):
            layer.update(dW, db, lr)


class AdamLayer:
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.t = 0
        self.m = None
        self.v = None

    def step(self, dW: np.ndarray, db: np.ndarray):
        if self.t == 0:
            self.m = np.zeros(dW.shape)
            self.v = np.zeros(db.shape)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * dW
        self.v = self.beta2 * self.v + (1 - self.beta2) * (dW ** 2)

        m_hat = self.m / (1 - (self.beta1 ** self.t))
        v_hat = self.v / (1 - (self.beta2 ** self.t))

        adam_dW = m_hat / (np.sqrt(v_hat) + self.epsilon)

        return adam_dW, db


class Adam(Optimizer):
    def __init__(self, layers: List[LinearLayer], adam_layers: List[AdamLayer] = []):
        super().__init__(layers)
        self.adam_layers = adam_layers

        if not adam_layers:
            self.adam_layers = [AdamLayer() for i in range(len(layers))]

    def update(self, weight_grads: List[np.ndarray], bias_grads: List[np.ndarray], lr: float) -> None:
        for layer, dW, db, adam in zip(self.layers, weight_grads, bias_grads, self.adam_layers):
            adam_dW, adam_db = adam.step(dW, db)
            layer.update(adam_dW, adam_db, lr)


"""
================= NeuralNet =========================
"""


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and performs classification
    over C classes. We train the network with a cross-entropy loss function and
    L2 regularization on the weight matrices.

    The network uses a nonlinearity after each fully connected layer except for
    the last. The outputs of the last fully-connected layer are passed through
    a softmax, and become the scores for each class."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        optimizer: Type[Optimizer] = SGD(),
        norm_weights: bool = False
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:

        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)

        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: The number of classes C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.norm_weights = norm_weights

        assert len(hidden_sizes) == (num_layers - 1)
        assert num_layers >= 1

        activated_layer_sizes = [input_size] + hidden_sizes
        activated_layers = [LinearLayer(n_in, n_out, activation_func=ReLU) for n_in, n_out in zip(activated_layer_sizes, activated_layer_sizes[1:])]
        final_layer = LinearLayer(activated_layer_sizes[-1], self.output_size, activation_func=Softmax)
        self.layers = activated_layers + [final_layer]

        self.optimizer = optimizer

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the scores for each class for all of the data samples.

        Hint: this function is also used for prediction.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample

        Returns:
            Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the last layer of your network
        """
        return reduce(lambda next_in, func: func(next_in), self.layers, X)

    def predict_from_scores(self, scores: np.ndarray) -> np.ndarray:
        """Compute the label based on the class scores for each item

        Parameters:
            scores: Results of forward pass (N, C)

        Returns:
            Matrix of shape (N, 1)
        """
        return np.argmax(scores, axis=1)

    def accuracy(self, scores: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict_from_scores(scores)

        correct = sum((y_pred == y_real for y_pred, y_real in zip(predictions, y)))

        return correct / len(y)

    def forward_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        scores = self.forward(X)
        return self.accuracy(scores, y)

    def one_hot_encode(self, y: np.ndarray) -> np.ndarray:
        """Convert input labels (N, 1) to one-hot encoding (N, C)

        Parameters:
            y: the actual labels (N, 1)

        Returns:
            The one-hot encoded labels
        """
        return np.eye(self.output_size)[y]

    def backward(
        self, X: np.ndarray, y: np.ndarray, lr: float, reg: float = 0.0
    ) -> float:
        """Perform back-propagation and update the parameters using the
        gradients.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training sample
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
            lr: Learning rate
            reg: Regularization strength

        Returns:
            Total cost for this batch of training samples
        """
        scores = self.forward(X)

        y_one_hot = self.one_hot_encode(y)
        loss = CrossEntropy.evaluate(y_one_hot, scores, reg, [layer.w for layer in self.layers])

        d_layer = CrossEntropy.evaluate_grad_with_softmax(y, scores)

        w_grads = []
        b_grads = []

        for idx, layer in reversed(list(enumerate(self.layers))):
            # Not output layer
            if (idx + 1) < len(self.layers):
                next_layer = self.layers[idx + 1]

                d_layer = d_layer.dot(next_layer.w.T)
                d_layer = layer.activation_func.evaluate_grad(d_layer, layer.activated_out)

            d_w = layer.linear_in.T.dot(d_layer) + 2 * reg * layer.w
            d_b = np.sum(d_layer, axis=0)

            w_grads.insert(0, d_w)
            b_grads.insert(0, d_b)

        self.optimizer.update(self.layers, w_grads, b_grads, lr)

        if self.norm_weights:
            w_norm = max(np.linalg.norm(l.w) for l in self.layers) / len(self.layers)
            b_norm = max(np.linalg.norm(l.w) for l in self.layers) / len(self.layers)
            for layer in self.layers:
                layer.w /= w_norm
                layer.b /= b_norm

        return loss
