import csv

from keras import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam

from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

raw_dataset = []
raw_answers = []

with open("wbcprbc_dataset.CSV", "r") as dataset_file:
    dataset_reader = csv.reader(dataset_file)


    for rbc, wbc, rbcprbc, label in dataset_reader:
        rbc, wbc, rbcprbc, label = float(rbc), float(wbc), float(rbcprbc), int(label)

        if wbc == 0:
            continue

        label = label - 1 if label > 0 else label

        raw_dataset.append(rbcprbc)
        raw_answers.append(label)

raw_dataset, raw_answers = np.array(raw_dataset), np.array(raw_answers)
x_train, x_test, y_train, y_test = train_test_split(raw_dataset, raw_answers, test_size=0.2, random_state=34)

model = Sequential([
    Dense(1, input_dim=1, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1024, activation='relu'),
    Dense(2, activation='sigmoid'),
])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=2, epochs=20, validation_data=(x_test, y_test))

plt.figure(figsize=(12, 4))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy Function')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper left')
plt.tight_layout()
plt.show()

res = model.predict(np.array([20.0]))

print(res)