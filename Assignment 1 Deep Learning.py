#!/usr/bin/env python
# coding: utf-8

# In[75]:


import getpass
import datetime
import socket

def generate_author_claim():
    # Get current user
    user = getpass.getuser()

    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get current IP address
    ip_address = socket.gethostbyname(socket.gethostname())

    # Enter your name
    name = input("Enter your full name: ")

    # Enter your email address
    email = input("Enter your email address: ")

    # Enter your student ID
    student_id = input("Enter your student ID: ")

    # Generate author claim string
    author_claim = f"Code authored by {user} ({name} {student_id} {email}) on {timestamp}  from IP address {ip_address}"

    return author_claim

# Generate the author claim string
author_claim = generate_author_claim()

# Print the author claim string
print(author_claim)


# In[76]:


# # Task 1.1 Understanding the Data
# # Describe the Fashion-MNIST Dataset
# # The Fashion-MNIST dataset is a collection of Zalando's article images, intended to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It consists of 70,000 grayscale images of 28x28 pixels each, with 60,000 images for training and 10,000 for testing. Each image is associated with a label from 10 classes:

# # T-shirt/top
# # Trouser
# # Pullover
# # Dress
# # Coat
# # Sandal
# # Shirt
# # Sneaker
# # Bag
# # Ankle boot

# Display 5 Training Examples from Each Target Class
# To visualize the data, we can use the following code snippet:

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

# Load Fashion-MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Display 5 training examples from each target class
fig, axes = plt.subplots(10, 5, figsize=(10, 15))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

for i in range(10):
    class_indices = np.where(train_labels == i)[0]
    selected_indices = np.random.choice(class_indices, 5, replace=False)
    for j, idx in enumerate(selected_indices):
        axes[i, j].imshow(train_images[idx], cmap='gray')
        axes[i, j].axis('off')
        if j == 0:
            axes[i, j].set_title(class_names[i])

plt.show()


# In[77]:


# Prepare the Data for Learning a Neural Network
# Preprocessing steps:

# Normalization: Scale pixel values to the range [0, 1] for better convergence.
# Reshape: Ensure the input data is in the correct shape for the neural network.
# Split the data: Create training, validation, and test datasets.
# See the code below:

from tensorflow.keras.utils import to_categorical

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape data
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Convert labels to one-hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Split the training data into training and validation datasets
validation_split = 0.1
num_val_samples = int(validation_split * train_images.shape[0])

val_images = train_images[:num_val_samples]
partial_train_images = train_images[num_val_samples:]

val_labels = train_labels[:num_val_samples]
partial_train_labels = train_labels[num_val_samples:]

print(f"Number of training examples: {partial_train_images.shape[0]}")
print(f"Number of validation examples: {val_images.shape[0]}")
print(f"Number of test examples: {test_images.shape[0]}")


# In[78]:


pip install pydot


# In[80]:


# Task 1.2 Setting up a Model for Training
# Model Configuration
# Output layer:

# Number of output nodes: 10 (one for each class)
# Activation function: Softmax
# Hidden layers:

# Number of hidden layers: 2 (this can be tuned)
# Number of nodes in each layer: 128 (this can be tuned)
# Activation function for each layer: ReLU
# Input layer:

# Input size: 784 (28x28 pixels)
# Reshape the input: Yes, to flatten the 28x28 images to a 1D array of 784 pixels
# Model Design Code

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import fashion_mnist
import os

# Ensure the 'dot' executable is in your PATH
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

# Load Fashion-MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Define the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Plot the model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# Display the model architecture
plt.figure(figsize=(10, 10))
img = plt.imread('model.png')
plt.imshow(img)
plt.axis('off')
plt.show()


# # Task 1.3: Fitting the Model
# Settings and Justifications
# Loss Function:
# 
# Categorical Crossentropy:
# Role: This is the most common loss function for multi-class classification problems. It measures the difference between the true label distribution and the predicted label distribution, providing a measure of how well the model's predictions match the true labels.
# Justification: The Fashion-MNIST dataset is a multi-class classification problem with 10 classes. Categorical Crossentropy is well-suited for this purpose as it helps in optimizing the model by minimizing the prediction error for multi-class labels.
# Metrics for Model Evaluation:
# 
# Accuracy:
# Role: Accuracy measures the proportion of correctly predicted labels out of the total labels. It provides a straightforward interpretation of how well the model is performing.
# Justification: Accuracy is an intuitive metric for classification tasks. For Fashion-MNIST, which involves distinguishing between different categories of clothing, accuracy gives a clear indication of the model's performance.
# Optimizer:
# 
# Adam Optimizer:
# Role: Adam (Adaptive Moment Estimation) combines the advantages of two other extensions of stochastic gradient descent. It computes adaptive learning rates for each parameter, making it well-suited for problems with sparse gradients and noisy data.
# Justification: Adam is known for its efficiency and good performance in a wide range of deep learning applications. It adjusts the learning rate during training, leading to faster convergence and better results.
# Training Batch Size:
# 
# Batch Size: 32
# Role: Batch size determines the number of samples that will be propagated through the network before updating the model parameters.
# Justification: A batch size of 32 is a common choice that balances training speed and model convergence stability.
# Number of Training Epochs:
# 
# Epochs: 20
# Role: The number of epochs determines how many times the entire training dataset will be passed through the network.
# Justification: 20 epochs are generally sufficient for the Fashion-MNIST dataset to achieve good performance without overfitting. This number can be adjusted based on the training and validation performance.
# Learning Rate:
# 
# Learning Rate: 0.001
# Role: The learning rate controls how much to change the model in response to the estimated error each time the model weights are updated.
# Justification: A learning rate of 0.001 is a standard starting point for the Adam optimizer. It is small enough to ensure stable convergence but large enough to make reasonable progress in each update.
# 
# 
# 
# Explanation of Training Stopping Criteria
# Early Stopping (if used): Early stopping is a technique used to prevent overfitting by monitoring the validation loss. Training stops if the validation loss does not improve for a certain number of epochs (patience).
# Manual Inspection: If early stopping is not used, manual inspection of the training and validation curves can help determine when to stop training. Look for the point where the validation loss starts to increase while the training loss continues to decrease, indicating overfitting.
# In this case, we'll monitor the training and validation loss and accuracy to decide if 20 epochs are sufficient or if adjustments are needed.

# In[81]:


# Model fitting
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.optimizers import Adam

# Load Fashion-MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model
history = model.fit(train_images, train_labels, epochs=20, batch_size=32, validation_data=(test_images, test_labels))

# Plot training & validation loss values
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.show()

Task 2.1: Check the Training Using Visualisation
To visualize the training process, we can use TensorBoard, a visualization toolkit for TensorFlow. TensorBoard allows us to monitor the training metrics, such as loss and accuracy, in real-time.

Step 1: Set Up TensorBoard Callback
First, we need to set up the TensorBoard callback to log the training metrics. 
# In[82]:


from tensorflow.keras.callbacks import TensorBoard
import datetime

# Define the log directory
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Fit the model with TensorBoard callback
history = model.fit(train_images, train_labels, epochs=20, batch_size=32, 
                    validation_data=(test_images, test_labels), callbacks=[tensorboard_callback])


# Step 2: Launch TensorBoard
# After setting up the callback and fitting the model, we can launch TensorBoard using the following command in the terminal:
# tensorboard --logdir=logs/fit
# This will start a TensorBoard server and provide a URL to access the visualizations in a web browser.
# 
# Step 3: Interpretation of the Visualizations
# By examining the TensorBoard visualizations, we can observe the training and validation loss and accuracy curves to detect overfitting or underfitting.
# 
# Overfitting: If the training loss continues to decrease while the validation loss starts to increase, it indicates overfitting.
# Underfitting: If both training and validation losses are high and do not decrease significantly, it indicates underfitting.
# 
# Task 2.2: Applying Regularisation
# To improve the training process and address overfitting, we can apply different regularization techniques such as Dropout, Batch Normalization, and L2 Regularization. We will compare their effects on model training.
# 
# Regularization Techniques
# 
# Dropout:
# Dropout randomly drops neurons during training to prevent the model from becoming too dependent on any specific neurons, thus reducing overfitting.
# 
# Batch Normalization:
# Batch Normalization normalizes the inputs of each layer to have a mean of 0 and a standard deviation of 1, which can help stabilize and accelerate training.
# 
# L2 Regularization:
# L2 Regularization adds a penalty term to the loss function proportional to the square of the weights, discouraging large weights and reducing overfitting.

# In[85]:


# Implementing Regularization Techniques
from tensorflow.keras.layers import Input

# Define the model with regularization
model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Set up TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Fit the model with TensorBoard callback
history = model.fit(train_images, train_labels, epochs=20, batch_size=32, 
                    validation_data=(test_images, test_labels), callbacks=[tensorboard_callback])



# Comparing Regularization Techniques
# By comparing the training and validation metrics with and without regularization, we can evaluate the effectiveness of each technique.
# 
# Dropout: Reduces overfitting by preventing the model from heavily relying on specific neurons.
# Batch Normalization: Helps in stabilizing and acceleratiion training.
# L2 Regularization: Discourages large weights thus reducing overfitting.
# 
# Task 2.3: Visualise the Trained Network
# After training the network with regularization, we can extract the output features and use the t-SNE algorithm to visualize them in 2D.

# In[90]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# This ensures the model is built
model.fit(train_images, train_labels, epochs=1, batch_size=32, verbose=1)


# In[4]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import umap
import matplotlib.pyplot as plt

# Initialize the model
model = Sequential()

# Add input layer
model.add(Flatten(input_shape=(28, 28, 1)))

# Add layers with Dropout and BatchNormalization
model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), name='dense1'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), name='dense2'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(10, activation='softmax', name='output_layer'))

# Define learning rate schedule
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=100,
    decay_rate=0.9
)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load dataset
(train_data, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
train_data = train_data / 255.0
train_data = train_data[..., np.newaxis]  # Add channel dimension

# Select a subset containing all classes
num_classes = 10
indices = np.concatenate([np.where(train_labels == i)[0][:100] for i in range(num_classes)])
train_data_sample = train_data[indices]
train_labels_sample = train_labels[indices]

# Train the model
model.fit(train_data_sample, train_labels_sample, epochs=5)

# Call the model on the sample data to ensure it's built
_ = model.predict(train_data_sample)

# Create models to extract embeddings from different layers
layer_outputs = [model.get_layer('dense1').output, model.get_layer('dense2').output]
feature_models = [Model(inputs=model.input, outputs=layer_output) for layer_output in layer_outputs]

# Extract features
features_dense1 = feature_models[0].predict(train_data_sample)
features_dense2 = feature_models[1].predict(train_data_sample)

# Define the UMAP plotting function
def plot_umap_representations(features, labels, layer_name):
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    umap_embeddings = umap_model.fit_transform(features)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, cmap='tab10', alpha=0.5)
    plt.colorbar(scatter, label='Class')
    plt.title(f'UMAP Visualization of {layer_name}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.show()

# Plot embeddings from different layers
plot_umap_representations(features_dense1, train_labels_sample, 'Dense Layer 1')
plot_umap_representations(features_dense2, train_labels_sample, 'Dense Layer 2')


# In[27]:


import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

# Load the Fashion MNIST dataset
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

# Normalize the pixel values to be between 0 and 1
train_data, test_data = train_data / 255.0, test_data / 255.0

# Add a channels dimension to the data
train_data = train_data[..., np.newaxis]
test_data = test_data[..., np.newaxis]

# Split the training data into training and validation sets
val_split = 0.1
val_size = int(len(train_data) * val_split)
val_data, val_labels = train_data[:val_size], train_labels[:val_size]
train_data, train_labels = train_data[val_size:], train_labels[val_size:]



# In[6]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
import os

# Define and compile the model
model = Sequential([
    Input(shape=(28, 28, 1)),  # Correctly specify input shape for image data
    Flatten(),                 # Flatten the 28x28 images to 1D
    Dense(128, activation='relu', name='dense1'),
    Dense(64, activation='relu', name='dense2'),
    Dense(10, activation='softmax', name='output_layer')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[7]:


# Create a directory for TensorBoard logs
log_dir = os.path.join("logs", "fit", "model")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


# In[8]:


history = model.fit(
    train_data, train_labels,
    epochs=10,
    validation_data=(val_data, val_labels),
    callbacks=[tensorboard_callback]
)


# In[10]:


tensorboard --logdir=logs/fit


# In[11]:


pip install tensorboard


# In[12]:


import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[13]:


# Define and compile the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu', name='dense1'),
    tf.keras.layers.Dense(64, activation='relu', name='dense2'),
    tf.keras.layers.Dense(10, activation='softmax', name='output_layer')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create dummy data
dummy_data = np.random.random((10, 28, 28, 1))
dummy_labels = np.random.randint(0, 10, size=(10,))

# Train the model on dummy data
model.fit(dummy_data, dummy_labels, epochs=1, batch_size=10)


# In[14]:


pip install umap-learn


# In[15]:


model.summary()


# Task 3: Analyze the Learned Representations
# To analyze the learned representations of your neural network, follow these steps to visualize and interpret the embeddings from different layers using UMAP.

# In[22]:


pip install umap-learn


# In[24]:


# # 1. Select a Subset of Training Data
# # Choose a representative subset of your training data that includes examples from all classes. 
# This subset should be large enough to capture the diversity of the data and to allow for meaningful visualization.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.initializers import GlorotUniform

# Define a simple model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Example input shape for Fashion-MNIST
    Dense(128, activation='relu', kernel_initializer=GlorotUniform(), name='dense_1'),
    Dense(64, activation='relu', kernel_initializer=GlorotUniform(), name='dense_2'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model (you might want to do this before extraction)
model.fit(x_train, y_train, epochs=5, validation_split=0.1)


# In[26]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.initializers import GlorotUniform

# Define a simple model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Example input shape for Fashion-MNIST
    Dense(128, activation='relu', kernel_initializer=GlorotUniform(), name='dense_1'),
    Dense(64, activation='relu', kernel_initializer=GlorotUniform(), name='dense_2'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_split=0.1)
 
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

# Define the functional API model
inputs = Input(shape=(28, 28))  # Example input shape for Fashion-MNIST
x = Flatten()(inputs)
x = Dense(128, activation='relu', kernel_initializer=GlorotUniform(), name='dense_1')(x)
x = Dense(64, activation='relu', kernel_initializer=GlorotUniform(), name='dense_2')(x)
outputs = Dense(10, activation='softmax')(x)

# Build and compile the model
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Define a model to output intermediate layer embeddings
layer_outputs = [layer.output for layer in model.layers if 'dense' in layer.name]  # Adjust filter if needed
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# Get embeddings for the subset
activations = activation_model.predict(x_subset)

import umap
import matplotlib.pyplot as plt

def plot_umap(embeddings, labels, layer_name):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    umap_embeddings = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, cmap='tab10', s=5)
    plt.colorbar(scatter, label='Class')
    plt.title(f'UMAP Visualization of Embeddings - {layer_name}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.show()

# Example: Plot for the first layer
layer_name = model.layers[0].name
plot_umap(activations[0], y_subset, layer_name)


# Task 4.1 Conceptual Understanding
# Implement Glorot Initialization in the Fashion-MNIST Classification Problem
# 
# Here’s an example of how you might implement Glorot Initialization in a neural network for Fashion-MNIST classification using TensorFlow/Keras:
# 

# In[17]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.initializers import GlorotUniform
import matplotlib.pyplot as plt

# Define the model with Glorot initialization
model_glorot = Sequential([
    Input(shape=(28, 28, 1)),
    Flatten(),
    Dense(128, activation='relu', kernel_initializer=GlorotUniform(), name='dense1'),
    Dense(64, activation='relu', kernel_initializer=GlorotUniform(), name='dense2'),
    Dense(10, activation='softmax', name='output_layer')
])

model_glorot.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess the dataset
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_data = train_data / 255.0
test_data = test_data / 255.0
train_data = train_data[..., np.newaxis]  # Add channel dimension
test_data = test_data[..., np.newaxis]    # Add channel dimension

# Train the model with Glorot initialization
history_glorot = model_glorot.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=10)

# Plot training history
plt.plot(history_glorot.history['accuracy'], label='Train Accuracy with Glorot Init')
plt.plot(history_glorot.history['val_accuracy'], label='Test Accuracy with Glorot Init')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Task 4.2 Implementation
# Implementation of Glorot Initialization in a neural network for Fashion-MNIST classification using TensorFlow/Keras:

# In[73]:


# Define the model without Glorot initialization (default initialization)
model_default = Sequential([
    Input(shape=(28, 28, 1)),
    Flatten(),
    Dense(128, activation='relu', name='dense1'),
    Dense(64, activation='relu', name='dense2'),
    Dense(10, activation='softmax', name='output_layer')
])

model_default.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with default initialization
history_default = model_default.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=10)

# Plot training history
plt.plot(history_default.history['accuracy'], label='Train Accuracy without Glorot Init')
plt.plot(history_default.history['val_accuracy'], label='Test Accuracy without Glorot Init')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:




