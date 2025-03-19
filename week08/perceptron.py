import numpy as np
import matplotlib.pyplot as plt


class Perceptron(object):
    # ==========================================#
    # The init method is called when an object #
    # is created. It can be used to initialize #
    # the attributes of the class.             #
    # ==========================================#
    def __init__(self, n_classes, n_features, max_iterations=20, learning_rate=0.1):
        self.n_classes = n_classes
        self.n_features = n_features
        self.weights = np.ones((n_classes, n_features + 1)) / (n_features + 1)
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

    # =======================================#
    # Prints the details of the perceptron. #
    # =======================================#
    def print_details(self):
        print("Number of classes:\t" + str(self.n_classes))
        print("Number of features:\t" + str(self.n_features))
        print("Max iterations:\t" + str(self.max_iterations))
        print("Learning rate:\t" + str(self.learning_rate))

    # =========================================#
    # Performs feed-forward prediction on one #
    # set of inputs.                          #
    # =========================================#
    def predict(self, inputs):
        scores = np.dot(inputs, self.weights.T)
        return np.argmax(scores)

    def _step(self, x):
        return 1 if x >= 0 else 0

    # ======================================#
    # Trains the perceptron using labelled #
    # training data.                       #
    # ======================================#
    def train(self, training_data, labels):
        assert len(training_data) == len(labels)
        # For each sample there is a label

        n_samples, n_features = training_data.shape

        # Add bias to the training data
        biased_data = np.c_[np.ones(n_samples), training_data]

        for _ in range(self.max_iterations):
            for i in range(n_samples):
                predicted_class = self.predict(biased_data[i])
                true_class = int(labels[i])

                if predicted_class != true_class:
                    self.weights[true_class] += self.learning_rate * \
                        biased_data[i]
                    self.weights[predicted_class] -= self.learning_rate * \
                        biased_data[i]

        return

    def batch_train(self, training_data, labels):
        assert len(training_data) == len(labels)

        n_samples, n_features = training_data.shape

        # Add bias to the training data
        biased_data = np.c_[np.ones(n_samples), training_data]

        for _ in range(self.max_iterations):
            # Initialize the gradient accumulator
            gradient = np.zeros_like(self.weights)

            # Accumulate the gradient for all samples
            for i in range(n_samples):
                predicted_class = self.predict(biased_data[i])
                true_class = int(labels[i])

                if predicted_class != true_class:
                    gradient[predicted_class] -= biased_data[i] * 1
                    gradient[true_class] += biased_data[i] * 1

            # Compute the average gradient
            gradient /= n_samples

            # Update the weights using the average gradient
            self.weights += self.learning_rate * gradient

        return

    def test_digit(self, testing_data, labels, digit):
        assert len(testing_data) == len(labels)

        # Filter samples where the true label matches the specified digit
        digit_indices = np.where(labels == digit)[0]
        digit_samples = testing_data[digit_indices]
        digit_labels = labels[digit_indices]

        digit_samples = np.c_[np.ones(len(digit_samples)), digit_samples]

        if len(digit_samples) == 0:
            print(f"No samples found for digit {digit}.")
            return 0.0

        accuracy = 0.0
        n_samples = len(digit_samples)

        for i in range(n_samples):
            # Predict the label for the current sample
            y_pred = self.predict(digit_samples[i])

            # Check if the prediction matches the true label
            accuracy += y_pred == digit_labels[i]

            # Print the prediction and true label
            print(
                f"Sample {i + 1}: Predicted = {y_pred}, Actual = {digit_labels[i]}")

        # Compute the accuracy for the specified digit
        accuracy /= n_samples
        print(f"Accuracy for digit {digit}: {accuracy:.2f}")

        return accuracy

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
            #            print("Predicted:\t" + str(y_pred) +
            #      "\tActual:\t" + str(labels[i]))
            accuracy += y_pred == labels[i]

        accuracy /= n_samples

        print("Accuracy:\t" + str(accuracy))


def visualize_weights_barplot(perceptron, class_idx):
    weights = perceptron.weights[class_idx]

    plt.figure(figsize=(28, 28))
    plt.imshow(weights[1:].reshape(28, 28))
    plt.title(f"Weights for Class {class_idx}")
    plt.xlabel("Features and Bias")
    plt.ylabel("Weight Value")
    plt.show()


def loadPreset(filePath):
    trainIn = np.loadtxt(filePath + "_train.csv", delimiter=",")
    testIn = np.loadtxt(filePath + "_test.csv", delimiter=",")
    return splitLabel(trainIn), splitLabel(testIn)


def splitLabel(array):
    return array[:, 0], array[:, 1:]
