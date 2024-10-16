import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class_names = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Display the shape of the training and test sets
print("Training data shape (images):", x_train.shape)
print("Training data shape (labels):", y_train.shape)
print("Test data shape (images):", x_test.shape)
print("Test data shape (labels):", y_test.shape)

# Number of training and test samples
num_train_samples = x_train.shape[0]
num_test_samples = x_test.shape[0]

print("Number of training samples:", num_train_samples)
print("Number of test samples:", num_test_samples)

# Shape of each image
image_shape = x_train.shape[1:]

print("Shape of each image:", image_shape)
print("Number of color channels:", image_shape[-1])

# Plot a few random images from the dataset
def plotImages(num_images_to_show,  dataset_samples):
    random_indices = np.random.choice(dataset_samples, num_images_to_show, replace=False)

    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(random_indices):
        plt.subplot(1, num_images_to_show, i + 1)
        plt.imshow(x_train[idx])
        plt.title(class_names[y_train[idx][0]])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

#plotImages(5, num_train_samples)
def normalize(x):
    x = x.astype('float32')
    x /= 255
    return x

X_train_norm = normalize(x_train)
Y_train_norm = normalize(y_train)
X_test_norm = normalize(x_test)
Y_test_norm = normalize(y_test)

def onehotVector(class_names):
    unique, inverse = np.unique(class_names, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot

print(onehotVector(class_names))

def barChart(dataset_label, title):
    data, counts = np.unique(dataset_label, return_counts=True)
    print(data, counts)
    plt.figure(figsize=(10, 5))
    plt.bar(class_names[data], counts)
    plt.title(title)
    plt.show()


barChart(y_train, 'Training data distribution')
barChart(y_test, 'Test data distribution')