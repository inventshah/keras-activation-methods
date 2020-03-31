import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# Save data sample
def save_sample(train_images, train_labels):
	class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
	plt.figure(figsize=(10,10))
	for i in range(25):
		plt.subplot(5,5,i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(train_images[i], cmap=plt.cm.binary)
		plt.xlabel(class_names[train_labels[i]])

	plt.savefig('images/data.png')
	plt.clf();

# Create model with activiation
def make_model(activation):
	model = keras.Sequential()
	model.add(keras.layers.Flatten(input_shape=(28, 28)))
	model.add(keras.layers.Dense(32, activation=activation))
	model.add(keras.layers.Dense(10))

	return model

# Train model
def train(model, train_images, train_labels, test_images, test_labels):
	model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
	return model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Save history graph
def save_history(history, name):
	plt.clf();
	plt.plot(history.history['accuracy'], label='accuracy')
	plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.ylim([0.5, 1])
	plt.legend(loc='lower right')
	plt.savefig("images/" + name + "_history.png")

# evaluate model accuracy
def evaluate(model, test_images, test_labels):
	test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
	print('\nTest accuracy:', test_acc)

def save_model(model, filename):
	model.save("models/" + filename + "_model.h5")

activation_functions = ["hard_sigmoid", "linear", "relu", "sigmoid", "softplus", "softsign", "tanh", "exponential"]

# Load data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

save_sample(train_images, train_labels)

for func in activation_functions:
	print("Running " + func)
	model = make_model(func);
	model.summary();
	quit();
	history = train(model, train_images, train_labels, test_images, test_labels)
	save_history(history, func)
	evaluate(model, test_images, test_labels)
	save_model(model, func);