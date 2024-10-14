import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
num_images_to_show = 5
random_indices = np.random.choice(num_train_samples, num_images_to_show, replace=False)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 5))
for i, idx in enumerate(random_indices):
    plt.subplot(1, num_images_to_show, i + 1)
    plt.imshow(x_train[idx])
    plt.title(class_names[y_train[idx][0]])
    plt.axis('off')

plt.tight_layout()
plt.show()
