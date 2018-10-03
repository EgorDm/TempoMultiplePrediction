import multiprocessing
import os

from keras.callbacks import ModelCheckpoint

import constants
import generators
import models
import numpy as np
import click
from keras.utils import to_categorical
import math


@click.command()
@click.option('--dataset', default='data/dataset_compound.npz', help='Dataset file path')
@click.option('--name', default='dnn_bpm_classify_real_comp', help='save path for dataset')
def main(dataset, name):
    with open(dataset, 'rb') as f:
        header = np.load(f)
        samples = np.load(f)

    samples = np.array([s for s in samples if 0 < s[2] <= 2])
    group_a = np.array([s for s in samples if s[2] == 1])
    group_b = np.array([s for s in samples if s[2] == 2])
    das_len = min(len(group_a), len(group_b))
    samples = np.concatenate((group_a[:das_len], group_b[:das_len]))

    xs = np.array(list(map(lambda v: v, samples[:, 0])))
    ys = np.array(list(map(lambda v: to_categorical(int(math.log2(v)), 2), samples[:, 2])))

    model = models.real_bpm_classification.create()
    model.summary()

    os.makedirs(constants.SAVE_PATH, exist_ok=True)
    with open(f'{constants.SAVE_PATH}/{name}_model.json', "w") as json_file: json_file.write(model.to_json())

    print('Train...')
    checkpoint = ModelCheckpoint(f'{constants.SAVE_PATH}/{name}_weights-{{epoch:02d}}-{{val_acc:.3f}}.h5', verbose=2, monitor='val_loss', save_best_only=True, mode='auto')
    model.fit(x=xs, y=ys, batch_size=200, epochs=30, verbose=2, validation_split=0.1, shuffle=True, callbacks=[checkpoint])

    model.save_weights(f'{constants.SAVE_PATH}/{name}_weights.h5')


if __name__ == "__main__":
    main()
