
import tensorflow as tf

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Display the shape of the training and test sets
print("Training data shape (images):", x_train.shape)
print("Training data shape (labels):", y_train.shape)
print("Test data shape (images):", x_test.shape)
print("Test data shape (labels):", y_test.shape)
