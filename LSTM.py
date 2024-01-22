import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
from tensorflow.keras import layers, initializers, regularizers
from sklearn.preprocessing import MinMaxScaler


# File upload, read the file content
###################################################
fname = os.path.join("sales.csv")
  
with open(fname) as f:
    data = f.read()
  
lines = data.split("\n")
header = lines[0].split(",")
lines = lines[1:]
lines = lines[: -1]

sales = np.zeros((len(lines),))
raw_data = np.zeros((len(lines), len(header) - 1)) 

for i, line in enumerate(lines):
    values = [float(x) for x in line.split(",")[1:]]
    sales[i] = values[0]                  
    raw_data[i, :] = values[0]
#####################################################


# Split train, validation, test data. Data preprocessing. 
#####################################################
num_train_samples = int(0.6 * len(raw_data))
num_val_samples = int(0.2 * len(raw_data))
num_test_samples = len(raw_data) - num_train_samples - num_val_samples

mean = raw_data[:num_train_samples].mean(axis=0)
raw_data -= mean
std = raw_data[:num_train_samples].std(axis=0)
raw_data /= std

mean_tem = sales[:num_train_samples].mean(axis=0)
sales -= mean_tem
std_mean = sales[:num_train_samples].std(axis=0)
sales /= std_mean
#####################################################


# Generate datasets that can be used for LSTM.
######################################################
sampling_rate = 1 
prediction_length = 1
sequence_length = 12
delay = sampling_rate * (sequence_length + prediction_length - 1)
batch_size = 1
  
train_dataset = keras.utils.timeseries_dataset_from_array(
    data = raw_data[:-delay],
    targets=sales[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples)

val_dataset = keras.utils.timeseries_dataset_from_array(
    data = raw_data[:-delay],
    targets=sales[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples)

test_dataset = keras.utils.timeseries_dataset_from_array(
    data = raw_data[:-delay],
    targets=sales[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples + num_val_samples,
    )
#############################################################


# Build the model.
#############################################################
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
# You can change hyperparameter and increase the accuracy
# The reason I didn't add more layers is less datasets. In general, if you have enough datasets you build model with deep layers.
# But if you build multi layer model without enough datasets, it cause overfitting.
# One of the way to prevent overfitting is to control hyperparameters such as regularization, dropout.
# The best way to prevent overfitting and to boost accuracy is more datasets.
# We can change optimizer, activation, initial value to boost accuracy.
# These are core concepts for deep learning, I am sorry with bad result.
# I tried my best for this project, but I think it's impossible with this datasets.
# I think traditional methods are better than deep learning with thsi datasets because deep learning requires enough datasets.
x = layers.LSTM(32, 
                recurrent_dropout=0.25, 
                activation="tanh", 
                kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.L2(1e-4),
                activity_regularizer=regularizers.L2(1e-5))(inputs)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="tanh")(x)
model = keras.Model(inputs, outputs)
##############################################################


# Train and testing
##############################################################
callbacks = [
    keras.callbacks.ModelCheckpoint("jena_dense.keras",          
                                    save_best_only=True)
] # Save the best model during training
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset,
                    epochs=5,
                    validation_data=val_dataset,
                    callbacks=callbacks)

model = keras.models.load_model("jena_dense.keras") # Load saved model
print(f"Test MAE: {model.evaluate(val_dataset)[0]:.2f}") # test the performance of the model

for inputs, targets in val_dataset:
    prediction = model.predict(inputs[:1])
    print(prediction * std + mean)

loss = history.history["mae"]
val_loss = history.history["val_mae"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training MAE")
plt.plot(epochs, val_loss, "b", label="Validation MAE")
plt.title("Training and validation MAE")
plt.legend()
plt.show()
##############################################################