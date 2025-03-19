import numpy as np
import matplotlib.pyplot as plt


class Perceptron(object):
    # ==========================================#
    # The init method is called when an object #
    # is created. It can be used to initialize #
    # the attributes of the class.             #
    # ==========================================#
    def __init__(self, no_inputs, max_iterations=20, learning_rate=0.1):
        self.no_inputs = no_inputs
        self.weights = np.ones(no_inputs + 1) / (no_inputs + 1)
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

    # =======================================#
    # Prints the details of the perceptron. #
    # =======================================#
    def print_details(self):
        print("No. inputs:\t" + str(self.no_inputs))
        print("Max iterations:\t" + str(self.max_iterations))
        print("Learning rate:\t" + str(self.learning_rate))

    # =========================================#
    # Performs feed-forward prediction on one #
    # set of inputs.                          #
    # =========================================#
    def predict(self, inputs):
        linear_output = np.dot(inputs, self.weights)
        return self._step(linear_output)

    def _step(self, x):
        return 1 if x >= 0 else 0

    # ======================================#
    # Trains the perceptron using labelled #
    # training data.                       #
    # ======================================#
    def train(self, training_data, labels):
        assert len(training_data) == len(labels)
        # For each sample there is a label

        # Weights are defined. len(Weights) is the same as the number of inputs
        n_samples, n_features = training_data.shape

        # Add bias to the training data
        biased_data = np.c_[np.ones(n_samples), training_data]

        for _ in range(self.max_iterations):
            for i in range(n_samples):
                print(self.weights)
                # Produce scalar, a weight * input product
                linear_output = np.dot(biased_data[i], self.weights)

                # Get the prediction
                y_pred = self._step(linear_output)

                # Update the weights
                update = self.learning_rate * (labels[i] - y_pred)
                self.weights += update * biased_data[i]

        return

    # =========================================mi#
    # Tests the prediction on each element of #
    # the testing data.
    # =========================================#
    def test(self, testing_data, labels):
        assert len(testing_data) == len(labels)
        accuracy = 0.0

        n_samples, n_features = testing_data.shape
        testing_data = np.c_[np.ones(n_samples), testing_data]

        for i in range(n_samples):
            y_pred = self.predict(testing_data[i])
            print("Predicted:\t" + str(y_pred) +
                  "\tActual:\t" + str(labels[i]))
            accuracy += y_pred == labels[i]

        accuracy /= n_samples

        print("Accuracy:\t" + str(accuracy))


def loadPreset(filePath):
    trainIn = np.loadtxt(filePath + "_train.csv", delimiter=",")
    testIn = np.loadtxt(filePath + "_test.csv", delimiter=",")
    return splitLabel(trainIn), splitLabel(testIn)


def splitLabel(array):
    return array[:, 0], array[:, 1:]
