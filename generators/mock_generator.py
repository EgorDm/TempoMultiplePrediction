from generators.base import BaseGenerator
import numpy as np
import random


class MockGenerator(BaseGenerator):
    def __init__(self, freq_range=600, ref_tempo=60):
        super().__init__()
        self.freq_range = freq_range
        self.ref_tempo = ref_tempo
        self.multiples = np.exp2([0, 1, 2, 3, 4, 5, 6]).astype(int)

    def get_sample(self):
        ref_pos = random.randint(self.ref_tempo, self.ref_tempo * 2 - 1)
        multiple = random.choice(self.multiples[:4])
        real_pos = ref_pos * multiple

        itempo = np.zeros(self.freq_range)
        hmultiples = np.zeros(self.freq_range)
        label = np.zeros(self.freq_range)

        intensity_distrib = [(0.25, (0.0, 0.15)), (0.5, (0.1, 0.3)), (1, (0.2, 0.5)), (2, (0.4, 1)), (4, (0.4, 0.9)), (8, (0.3, 0.7))]
        for m, dist in intensity_distrib:
            intensity = np.random.uniform(*dist)
            peak_width = random.randint(4, 18)
            distrib = np.hanning(peak_width)
            distrib = distrib / np.sum(distrib) * intensity
            pos = int(real_pos * m)

            real_start = int(pos - (np.rint(peak_width / 2)))
            start = int(max(0, real_start))
            end = int(min(self.freq_range - 1, real_start + peak_width))

            norm = distrib[start - real_start:end - real_start]
            itempo[start:end] = norm

        for m in self.multiples:
            pos = ref_pos * m
            if pos not in range(self.freq_range): break
            hmultiples[pos] = 1 if m == 1 else 0.5
            if m == multiple: label[pos] = 1

        return np.stack((itempo / np.sum(itempo), hmultiples), 1), label
