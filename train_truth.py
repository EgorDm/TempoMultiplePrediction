import multiprocessing
import os

from keras.callbacks import ModelCheckpoint

import constants
import generators
import models
import numpy as np
import click
from keras.utils import to_categorical


@click.command()
@click.option('--dataset', default='data/dataset.npz', help='Dataset file path')
@click.option('--name', default='dnn_bpm_classify_real', help='save path for dataset')
def main(dataset, name):
    with open(dataset, 'rb') as f:
        header = np.load(f)
        samples = np.load(f)

    xs = np.array(list(map(lambda v: v, samples[:, 0])))
    ys = np.array(list(map(lambda v: to_categorical(v - 1, 4), samples[:, 2])))

    model = models.real_bpm_classification.create()
    model.summary()

    os.makedirs(constants.SAVE_PATH, exist_ok=True)
    with open(f'{constants.SAVE_PATH}/{name}_model.json', "w") as json_file: json_file.write(model.to_json())

    print('Train...')
    checkpoint = ModelCheckpoint(f'{constants.SAVE_PATH}/{name}_weights-{{epoch:03d}}.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
    model.fit(x=xs, y=ys, batch_size=200, epochs=100, verbose=1, validation_split=0.04, shuffle=True, callbacks=[checkpoint])

    model.save_weights(f'{constants.SAVE_PATH}/{name}_weights.h5')


if __name__ == "__main__":
    main()
