import multiprocessing

import generators
import models

generator = generators.MockGenerator()
model = models.real_bpm_classification.create()

print('Train...')
model.fit_generator(
    generator=generator.get_generator(50),
    steps_per_epoch=400,
    epochs=10,
    verbose=2,
    validation_data=generator.get_generator(50),
    validation_steps=100,
    workers=multiprocessing.cpu_count(),
)

