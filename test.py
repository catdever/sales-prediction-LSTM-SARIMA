import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.callbacks import EarlyStopping

data = pd.read_csv('sale_new.csv')

data = data.drop("year", axis=1)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences(data, sequence_length):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i+sequence_length])
        print(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
        print(data[i+sequence_length])
    return np.array(x), np.array(y)

sequence_length = 6  # Anzahl der Monate, die als Eingabe verwendet werden sollen

print("data scaled ================================> ", data_scaled)
print("data scaled inversed ================================> ", scaler.inverse_transform(data_scaled))

x_train, y_train = create_sequences(data_scaled, sequence_length)

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
model.add(Dense(1))
val_accuracy_metric = Accuracy(name = 'val_accuracy')
model.compile(optimizer='adam', loss='mse', metrics=['accuracy', val_accuracy_metric])

x_val = data_scaled[-sequence_length:]
y_val = data_scaled[-sequence_length:]

early_stopping = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)

model.fit(x_train, y_train, epochs=200, batch_size=12, validation_data=(x_val, y_val), callbacks=[early_stopping], verbose=1)

predictedvalues = []
predicted_values = []

for i in range(12):
    last_sequence = data_scaled[-sequence_length:].reshape(1, sequence_length, 1)
    predicted_value = model.predict(last_sequence)
    predicted_values.append(scaler.inverse_transform(predicted_value)[0][0])
    data_scaled = np.append(data_scaled, predicted_value)

start_date = pd.to_datetime("2019-01-01")
months_2019 = [start_date + pd.DateOffset(months=i) for i in range(12)]

plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Wert'], label='Historische Daten von 2012-2019', color='blue')

x_values = np.arange(len(data), len(data) + 12)  # X-Werte für 12 Monate in 2019
plt.plot(x_values, predicted_values, marker='o', color='red', label='Prognostizierte Werte für 2019')

plt.xlabel('Monat')
plt.ylabel('Wert')
plt.legend()
plt.title('Historische Daten und Prognostizierte Werte für die Jahre 2012-2019')
plt.show()

print("x_train =====================> ", x_train)
print("y_train =====================> ", y_train)
print("x_val =====================> ", x_val)
print("x_val =====================> ", y_val)





# int_sequence = np.arange(20)                                
# dummy_dataset = keras.utils.timeseries_dataset_from_array(
#     data=int_sequence[:-6],                                 
#     targets=int_sequence[6:],  
#     sampling_rate=2,                             
#     sequence_length=3, 
#     shuffle=True,                                     
#     batch_size=2,
#     start_index=1                                           
# )

# for inputs, targets in dummy_dataset:
#     print(inputs, " : ", targets)
#     print(inputs.shape, " : ", targets.shape)
#     # for i in range(inputs.shape[0]):
#     #     print([int(x) for x in inputs[i]], int(targets[i]))