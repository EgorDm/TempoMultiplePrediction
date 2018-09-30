from keras import Sequential, Input
from keras.layers import Flatten, Dense


def create(n_classes=600):
    hidden_size_1 = int(n_classes/4)
    hidden_size_2 = int(n_classes/8)
    hidden_size_3 = int(n_classes/16)
    result_classes = 2

    model = Sequential()
    model.add(Flatten(input_shape=(n_classes, 2)))
    model.add(Dense(hidden_size_1, activation='relu'))
    model.add(Dense(hidden_size_2, activation='relu'))
    model.add(Dense(hidden_size_3, activation='relu'))
    model.add(Dense(result_classes, activation='sigmoid'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

