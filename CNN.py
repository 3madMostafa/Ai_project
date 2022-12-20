from tensorflow.keras import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Read the data...
data = pd.read_csv(r"C:\\Users\\top\\Downloads\\emnist-letters\\emnist-letters-train.csv").astype('float32')

# Split data the X - Our data , and y - the prdict label
X = data.drop('0', axis=1)
y = data['0']

# Reshaping the data in csv file so that it can be displayed as an image...

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
train_x = np.reshape(train_x.values, (train_x.shape[0], 28, 28))
test_x = np.reshape(test_x.values, (test_x.shape[0], 28, 28))

# Dictionary for getting characters from index values...
word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',
             11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
             21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

#Shuffling the data ...
shuff = shuffle(train_x[:100])

#Reshaping the training & test dataset so that it can be put in the model...

train_X = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
test_X = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)

# Converting the labels to categorical values...

train_y = to_categorical(train_y, num_classes=26, dtype='int')
test_y = to_categorical(test_y, num_classes=26, dtype='int')

# CNN model

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(26, activation="softmax"))
model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

history = model.fit(train_X, train_y, epochs=1, callbacks=[early_stop],  validation_data=(test_X, test_y))

model.summary()
model.save(r'model_hand.h5')

#Making model predictions...

pred = model.predict(test_X[:9])
