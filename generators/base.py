import numpy as np


class BaseGenerator:
    def get_sample(self): return None, None

    def get_batch(self, n):
        inputs, labels = [], []
        for _ in range(n):
            i, l = self.get_sample()
            inputs.append(i)
            labels.append(l)

        return np.array(inputs), np.array(labels)

    def get_generator(self, n):
        while True:
            yield self.get_batch(n)
