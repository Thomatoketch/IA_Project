import numpy as np
import tensorflow as tf
import math, pickle, sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras import layers
import keras

# ------------ Parameters -----------#
DATASET = 2
# 2 : DataSet "MNIST" in tf.keras   :  (nSamples = 60000 , nFeatures = 784 = 28x28)

learningRate = 0.01
maxIterations = 5

nHidden = 128 # ??#       #Number of neurones in hidden layer
filter = 32 # ??#       #Number of filters in convolution layer
ConvKernel = (3,3) # ??#       #Size of filters in convolution layer
Poolkernel = (2,2) # ??#       #Size of filters in pooling layer


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
    # Sélectionner une image aléatoire
    index = np.random.randint(0, len(images))

    # Détermination du nom du dataset (ici CIFAR-10)
    name = "CIFAR10"
    cmap = None  # Pas de colormap pour les images RGB
    image = images[index]  # Les images sont déjà en forme (32, 32, 3)

    # Liste des classes pour CIFAR-10
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # Convertir les étiquettes one-hot en leur classe correspondante
    label = classes[np.argmax(labels[index])]  # Décoder l'étiquette réelle
    predicted = classes[predictions[index]]  # Classe prédite

    # Définir le titre et le nom de fichier pour sauvegarder l'image
    title = f'CNN {name}: Label: {label}, Predicted: {predicted}'
    file_name = f"./{name}/CNN_{name}_{label}_{index}.pdf"

    # Afficher l'image avec son étiquette et sa prédiction
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.grid(False)
    plt.axis('off')  # Désactiver les axes

    # Sauvegarder l'image en PDF
    plt.show()  # Afficher l'image à l'écran


def plot_history(history, model):
    """
    Displays 'Cross Entropy loss' for Training and Testing set, for each iteration. In the same figure.
    Displays 'Accuracy' for Training and Testing set, for each iteration. In a second figure.
    """
    # Récupérer les valeurs de l'historique
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Figure 1 : Evolution de la perte
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title(f'{model}: Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    # Figure 2 : Evolution de l'exactitude
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title(f'{model}: Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


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
        self.n_hidden = nHidden
        self.n_filter = filter
        self.n_iterations = maxIterations
        self.learning_rate = learningRate
        self.hidden_activation = ReLU()  # To fix as 'ReLU' or 'Sigmoid'
        self.output_activation = Softmax()  # For classifiction models
        self.loss = CrossEntropy()
        self.Ckernel = ConvKernel
        self.Pkernel = Poolkernel


# ---------------------- CNN_VGG1: with Keras in TensorFlow ------------------#
def Keras_CNN_VGG1(cnn, X_train, y_train, X_test, y_test, opt="SGD"):
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
    shapeIn = (32, 32, 3)

    # 1- Creating CNN1 Model : Architecture with one VGG Block
    model = Sequential()

    # Layer 1: Convolutional Layer with 6 filters, 5x5 kernel size
    model.add(layers.Conv2D(cnn.n_filter, cnn.Ckernel, padding="same", activation='relu', input_shape=shapeIn))

    # Layer 2: Convolutional Layer with 16 filters, 5x5 kernel size
    model.add(layers.Conv2D(cnn.n_filter, cnn.Ckernel, padding="same", activation='relu'))

    # Layer 3: Max Pooling Layer
    model.add(layers.MaxPooling2D(cnn.Pkernel))

    # Layer 1: Convolutional Layer with 6 filters, 5x5 kernel size
    model.add(layers.Conv2D(cnn.n_filter, cnn.Ckernel, padding="same", activation='relu', input_shape=shapeIn))

    # Layer 2: Convolutional Layer with 16 filters, 5x5 kernel size
    model.add(layers.Conv2D(cnn.n_filter, cnn.Ckernel, padding="same", activation='relu'))

    # Layer 3: Max Pooling Layer
    model.add(layers.MaxPooling2D(cnn.Pkernel))

    # Layer 4: Flatten the output from the previous layer
    model.add(layers.Flatten())

    # Layer 5: Fully Connected Layer
    model.add(layers.Dense(cnn.n_hidden, activation='relu'))

    # Layer 6: Output Layer with softmax activation for 10 classes (for CIFAR-10, MNIST, etc.)
    model.add(layers.Dense(10, activation="softmax"))

    # 2- Fixing Optimizer algorithm and error function
    # SGD  - Stochastic Gradient Descent
    # Adam - adapts itself the learning Rate !
    if opt == "SGD":
        optimizer = keras.optimizers.SGD()
    else:
        optimizer = keras.optimizers.Adam()

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # 3- Training model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # 4- Testing model
    test_loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f'Test accuracy: {accuracy}')

    # 5- Get predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)

    # 6- Call plot_image function
    for i in range(maxIterations):
        plot_image(X_test, y_test, predicted_classes)

    # 7- Call plot_history to show loss and accuracy
    plot_history(history, "CNN_VGG1")

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

    # Charger les données CIFAR10
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalisation des données pour avoir des valeurs entre 0 et 1
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Convertir les étiquettes en vecteurs catégoriels
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test,10)

    # Créer une instance de la classe ConvolutionNeuralNetwork
    cnn = ConvolutionNeuralNetwork()

    # Lancer le modèle Keras_CNN_VGG1
    accuracy_ADAM = Keras_CNN_VGG1(cnn, X_train, y_train, X_test, y_test, opt="ADAM")
    accuracy_SGD = Keras_CNN_VGG1(cnn, X_train, y_train, X_test, y_test, opt="SGD")
    print(f'Model accuracy for Adam: {accuracy_ADAM}')
    print(f'Model accuracy for SGD: {accuracy_SGD}')
