import os
import numpy as np
import matplotlib.pyplot as plt

import generators
import models

name = 'dnn_bpm_class'
model = models.load_model(name)
if model is None: raise Exception(f'No model {name} found')

generator = generators.MockGenerator()


def eval_sample():
    multiples = [1, 2, 4, 8]
    x, y, r = generator.get_sample_eval()
    p = model.predict(np.array([x]), 1)[0]
    mp = multiples[int(np.argmax(p))]
    my = multiples[int(np.argmax(y))]
    scale = max(x[:, 0])

    plt.plot(x[:, 1] * scale, label='Multiples')

    mps = np.zeros(x.shape[0])
    mps[int(mp * r)] = 1 * scale
    mys = np.zeros(x.shape[0])
    mys[int(my * r)] = 0.9 * scale

    plt.plot(mps, label='Predicted')
    plt.plot(mys, label='Ground truth')

    plt.plot(x[:, 0], label='Histogram')
    plt.legend()
    plt.show()

for i in range(6): eval_sample()

