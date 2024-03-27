import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

keras = tf.keras

from keras.preprocessing.text import Tokenizer
from keras import regularizers
from keras.callbacks import EarlyStopping

# from keras.preprocessing.sequence import pad_sequences


# Initialize some parameters for tokenization, embedding and training
vocab_size = 5000
embedding_dim = 16
max_length = 100
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"
training_size = 1025


# Load data from json datastore
with open("bin_reviews.json", "r") as f:
    datastore = json.load(f)

comments = []
ratings = []

for item in datastore:
    comments.append(item["comment"])
    ratings.append(item["rating"])


# Split into training and testing
training_comments = comments[:training_size]
training_ratings = ratings[:training_size]
testing_comments = comments[training_size:]
testing_ratings = ratings[training_size:]


# Tokenize the words in testing comments
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_comments)

word_index = tokenizer.word_index


# Convert the training and testing comments to padded sequences of the word index indices
training_sequences = tokenizer.texts_to_sequences(training_comments)
training_padded = keras.preprocessing.sequence.pad_sequences(
    training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
)

testing_sequences = tokenizer.texts_to_sequences(testing_comments)
testing_padded = keras.preprocessing.sequence.pad_sequences(
    testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
)

training_padded = np.array(training_padded)
training_labels = np.array(training_ratings)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_ratings)


# Build a Sequential Neural Network
model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(
            8, activation="relu", kernel_regularizer=regularizers.l2(0.01)
        ),
        tf.keras.layers.Dense(
            1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01)
        ),
    ]
)


initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True
)


early_stopping = EarlyStopping(
    monitor="val_accuracy",  # Metric to monitor
    patience=5,  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored metric
)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()


# Train the model using 30 epochs and then validate
num_epochs = 100
history = model.fit(
    training_padded,
    training_labels,
    epochs=num_epochs,
    validation_data=(testing_padded, testing_labels),
    verbose=2,
)


# Plot the model accuracy and validation
def plot_graphs(history, string, filename):
    plt.plot(history.history[string])
    plt.plot(history.history["val_" + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, "val_" + string])
    plt.savefig(filename)
    plt.show()


plot_graphs(history, "accuracy", "bin_accuracy_plot.png")
plot_graphs(history, "loss", "bin_loss_plot.png")

sentence = [
    "Willing to help others, very patient and kind to students",
    "They presented slides during lab time with extremely helpful examples for completing coursework.",
    "Approachable and easy to talk to",
    "Asking questions",
]
sequences = tokenizer.texts_to_sequences(sentence)
padded = keras.preprocessing.sequence.pad_sequences(
    sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
)
print(model.predict(padded))
