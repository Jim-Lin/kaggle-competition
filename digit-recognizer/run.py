import numpy as np
import pandas as pd
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('train.csv')
X = df.drop(['label'], axis=1).as_matrix()
y = to_categorical(df['label'])
X_test = pd.read_csv('test.csv').as_matrix()

model = Sequential()
model.add(Dense(25, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(25, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, validation_split=0.3, epochs=1)

predict = np.argmax(model.predict(X_test), axis=1)

result = pd.DataFrame({'ImageId': np.arange(1, predict.shape[0]+1), 'Label': predict})
result.to_csv('submission.csv', index=False)
