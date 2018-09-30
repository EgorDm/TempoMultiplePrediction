import math
import multiprocessing
import os

import keras
from keras.callbacks import ModelCheckpoint

import constants
import generators
import models
import numpy as np
import click
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import random


def eval_sample(x, y, r, model: keras.Model, start=30):
    multiples = [1, 2, 4, 8]
    p = model.predict(np.array([x]), 1)[0]
    mp = multiples[int(np.argmax(p))]
    my = multiples[int(np.argmax(y))]
    scale = max(x[:, 0])

    plt.plot(x[:, 1] * scale, label='Multiples')
    plt.plot(x[:, 0], label='Histogram')

    mps_r = int(mp * r - start)
    if mps_r < len(x):
        mps = np.zeros(x.shape[0])
        mps[mps_r] = 1 * scale
        plt.plot(mps, label='Predicted')

    mys_r = int(my * r - start)
    if mys_r < len(x):
        mys = np.zeros(x.shape[0])
        mys[mys_r] = 0.8 * scale
        plt.plot(mys, label='Ground truth')

    plt.legend()
    plt.show()


@click.command()
@click.option('--dataset', default='data/dataset_compound.npz', help='Dataset file path')
@click.option('--model', default='dnn_bpm_classify_real_comp_model', help='Model save file')
@click.option('--weights', default='dnn_bpm_classify_real_comp_weights-30-0.885', help='Weigth save file')
@click.option('--count', default=10, help='Count')
def main(dataset, model, weights, count):
    with open(dataset, 'rb') as f:
        header = np.load(f)
        samples = np.load(f)

    samples = np.array([s for s in samples if 0 < s[2] <= 2])

    xs = np.array(list(map(lambda v: v, samples[:, 0])))
    ys = np.array(list(map(lambda v: to_categorical(int(math.log2(v)), 2), samples[:, 2])))

    model = models.load_model(model, weights)

    for _ in range(count):
        randi = random.randint(0, len(samples) - 1)
        eval_sample(xs[randi], ys[randi], samples[randi, 1], model)


if __name__ == "__main__":
    main()
