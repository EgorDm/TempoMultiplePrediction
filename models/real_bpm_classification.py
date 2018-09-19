from keras import Sequential, Input
from keras.layers import Flatten, Dense


def create(n_classes=600):
    hidden_size = int(n_classes/4)
    code_size = int(n_classes/8)

    model = Sequential()
    model.add(Input(shape=(None, n_classes, 2)))
    model.add(Flatten())
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(code_size, activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(n_classes, activation='relu'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', 'loss'])

    return model

