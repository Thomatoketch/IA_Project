import numpy as np
import tensorflow as tf
import math, pickle, sys
import matplotlib.pyplot as plt
from sklearn import datasets as datasetsSk
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

# ------------ Parameters -----------#
DATASET = 2
# 1 : DataSet sklearn.load_digits() :  (nSamples = 1797  , nFeatures = 64  = 8x8)
# 2 : DataSet "MNIST" in tf.keras   :  (nSamples = 60000 , nFeatures = 784 = 28x28)

learningRate = 0.01
maxIterations = 5

nHidden1 =  # ??#       #Number of neurones in hidden layer 1
nHidden2 =  # ??#       #Number of neurones in hidden layer 2
ConvKernel =  # ??#       #Size of filters in convolution layer
Poolkernel =  # ??#       #Size of filters in pooling layer


# ---------- Helpers Functions  -------------#

def normalize(X, axis=-1, order=2):
    ''' Normalize the dataset X
    -Each vector ligne x (an entry of X) is normalized as x = (x / ||x||_2 )
    -axis=-1: normalisation doit être appliquée horizontalement le long de chaque ligne
    -order=2: euclidean norm
    '''

    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


def to_categorical(x, n_col=None):
    ''' One-hot encoding of nominal values
    for all element in x in {0,...,9}
    Example : 0 will be encoded as [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Example : 1 will be encoded as [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    Example : 9 will be encoded as [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    '''

    if not n_col:
        n_col = np.max(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1  # multiple affectation
    return one_hot


def plot_image(images, labels, predictions):
    '''Displays a random image in 'test' dataset its label, and predicted value '''
    # Détermination de la taille de l'image basée sur la longueur du vecteur d'image

    index = np.random.randint(0, len(images))

    # Determine settings based on the dataset
    if DATASET == 1 or DATASET == 2:
        name = "LOAD_DIGITS" if DATASET == 1 else "MNIST"
        img_shape = (8, 8) if DATASET == 1 else (28, 28)
        cmap = 'gray'
        image = images[index].reshape(img_shape)  # Reshape grayscale images
        title = f'CNN {name}: Label: {labels[index]}, Predicted: {predictions[index]}'
        file_name = f"./{name}/CNN_{name}_{labels[index]}_{index}.pdf"
    elif DATASET == 3:
        name = "CIFAR10"
        # No Reshape, already in  (10000, 3, 32, 32)
        cmap = None  # No color map needed for RGB
        image = images[index]  # Ensure image is correctly shaped
        classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        # Assuming labels and predictions are already decoded to category names:
        title = f'CNN {name}: Label: {classes[labels[index]]}, Predicted: {classes[predictions[index]]}'
        file_name = f"./{name}/CNN_{name}_{classes[labels[index]]}_{index}.pdf"

    # Plot the image
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.grid(False)
    plt.axis('off')  # Turn off axis

    # Save the plot as a PDF file
    plt.savefig(file_name)
    # plt.show()  # Display the image plot


def plot_history(history, model):
    """
    Displays 'Cross Entropy loss' for Training and Testing set, for each iteration. In the same figure.
    Displays 'Accuracy' for Training and Testing set, for each iteration. In a second figure.
    """


# --------- Cross Entropy Error Class  -------------#
class CrossEntropy:
    def __init__(self): pass

    def loss(self, y, p):
        '''Cross-Entropy Loss function for multiclass predictions'''
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -np.sum(y * np.log(p))

    def acc(self, y, p):
        ''' Accuracy between One-hot encoding : target value 'y' and predicted 'p' '''
        # np.argmax translates to nominal values, for each entry.
        # the whole values are given to %accuracy function
        return accuracy(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        '''Gradient of Cross-Entropy function with respect to the input of softmax, not the softmax output itself'''
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return p - y  # This is the gradient for the input to softmax when using cross-entropy loss


# --------- Sigmoid activation Class : hidden layers  -------------#
class Sigmoid():
    def __call__(self, x):
        '''Sigmoid function'''
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        '''Derivative of Sigmoid function'''
        return self.__call__(x) * (1 - self.__call__(x))


# --------- ReLu activation Class : hidden layers  -------------#
class ReLU():
    def __call__(self, x):
        '''ReLU activation function'''
        return np.maximum(0, x)

    def gradient(self, x):
        '''Derivative of the ReLU function'''
        return 1. * (x > 0)


# --------- Softmax activation Class : output layer  --------------#
class Softmax():
    def __call__(self, x):
        '''Softmax function'''
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        '''Derivative of Softmax function'''
        p = self.__call__(x)
        return p * (1 - p)


# ---------------------- CNN parameters -------------------------------#
class ConvolutionNeuralNetwork():
    '''
    Parameters to use for LeNet5 CNN architecture
    '''

    def __init__(self):
        '''Initialization of CNN "hyper-parameters" '''
        self.n_hidden1 = nHidden1
        self.n_hidden2 = nHidden2
        self.n_iterations = maxIterations
        self.learning_rate = learningRate
        self.hidden_activation = ReLU()  # To fix as 'ReLU' or 'Sigmoid'
        self.output_activation = Softmax()  # For classifiction models
        self.loss = CrossEntropy()
        self.Ckernel = ConvKernel
        self.Pkernel = Poolkernel


# ---------------------- CNN_LeNet5: with Keras in TensorFlow ------------------#
def Keras_CNN_LeNet5(cnn, X_train, y_train, X_test, y_test, opt="SGD"):
    ''' Using TensorFlow library
    1- Create LeNet5 CNN model with tf.keras
    2- Fix algorithm optimizer (SGD, Adam) and error function
    3- Train the model
    4- Test the model
    5- Plot graphics
    '''

    h_activation = type(cnn.hidden_activation).__name__.lower()
    out_activation = type(cnn.output_activation).__name__.lower()

    # To use CNN example: Reshape datasets from flattered (50000, 3072) to (50000, 32, 32, 3)
    if DATASET == 1:
        X_train = X_train.reshape(-1, 8, 8, 1)  # One channel dimension (i.e. one color channel, the gray one)
        X_test = X_test.reshape(-1, 8, 8, 1)  # One  channel dimension (i.e. one color channel, the gray one)
        shapeIn = (8, 8, 1)
    if DATASET == 2:
        X_train = X_train.reshape(-1, 28, 28, 1)  # One  channel dimension (i.e. one color channel, the gray one)
        X_test = X_test.reshape(-1, 28, 28, 1)  # One  channel dimension (i.e. one color channel, the gray one)
        shapeIn = (28, 28, 1)

    # 1- Creating CNN1 Model : Architecture with one VGG Block

    # 2- Fixing Optimizer algorithm and error function
    # SGD  - Stochastic Gradient Descent
    # Adam - adapts itself the learning Rate !

    # 3- Training model

    # 4- Testing model

    # 5- Get predictions

    # 6- Call plot_image function
    for _ in range(5):
        plot_image(X_test, y_test, predicted_classes)

    # 7- Call plot_history to show loss and accuracy
    plot_history(history, "CNN_LeNet5")

    return accuracy


if __name__ == "__main__":
    """
    gpu_info = !nvidia-smi
    gpu_info = '\n'.join(gpu_info)
    if gpu_info.find('failed') >= 0:
        print('Not connected to a GPU')
    else:
        print(gpu_info)
    """

    ###################  1- Importing DataSet ###############
    if DATASET == 1:
        # Data : sklearn.load_digits()
        # Size : (1797 , 64 = 8x8)
        data = datasetsSk.load_digits()
        X, y = normalize(data.data), data.target

        # Convert the nominal y values to binary
        y = to_categorical(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
        print("Training      : ", X_train.shape)
        print("Test          : ", X_test.shape)

    if DATASET == 2:
        # Data : https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
        # Size : (60000 , 784 = 28x28)

        # Charger les données MNIST
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # Normalisation des données pour avoir des valeurs entre 0 et 1
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        # Redimensionner les images de (60000, 28, 28) à (60000, 784)
        X_train = X_train.reshape(X_train.shape[0], 28 * 28)
        X_test = X_test.reshape(X_test.shape[0], 28 * 28)

        print("Training      : ", X_train.shape)
        print("Test          : ", X_test.shape)

        # Convertir les étiquettes en vecteurs catégoriels
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

    ##################### 2- Creating Model #################

    #################### 3- CNN with TensorFlow ##############


