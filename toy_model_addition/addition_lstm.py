# An RNN that learns arithmatic addition
# generate_addition_data.py contains the how the data is genereated
# epoch 200 loss: 3.0851e-04 - acc: 1.0000 - val_loss: 0.8984 - val_acc: 0.8407

from __future__ import print_function
from keras.models import Sequential
from keras import layers
import numpy as np
import generate_addition_data

DATA_SIZE = 10000 # This includes both training and validation
MAX_DIGITS = 3 # The maxmimum number of digits for input numbers a and b
RNN = layers.LSTM
HIDDEN_SIZE = 128 # Size of encoders' output
BATCH_SIZE = 128
LAYERS = 1 # number of LSTM layers used for decoder
alphabet = '1234567890+ '

data_generator = generate_addition_data.AdditionDataGenerator(MAX_DIGITS, DATA_SIZE, alphabet)
x_train, y_train, x_vali, y_vali = data_generator.generate()

print('Data generation completed!')
print('x_train.shape', x_train.shape)
print('y_train.shape', y_train.shape)
print('x_vali.shape', x_vali.shape)
print('y_vali.shape', y_vali.shape)

# The shape for data is (example_number, char_number, which_char_from_alphabet)

print('Building RNN model...')
model = Sequential() # A stack of Keras layers
model.add(RNN(HIDDEN_SIZE, input_shape=(MAX_DIGITS * 2 + 1, len(alphabet)))) # Encoder encodes each digit to a hiddensize vector
model.add(layers.RepeatVector(MAX_DIGITS + 1)) # Now output becomes (batch_size, MAX_DIGITS+1, HiddenSize)
# The decoder RNN can have multiple layers
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

model.add(layers.TimeDistributed(layers.Dense(len(alphabet))))
model.add(layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

print('Training RNN model ...')
for epoch in range(1, 400):
    print('-' * 50)
    print('We are on epoch: ', epoch)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(x_vali, y_vali))
