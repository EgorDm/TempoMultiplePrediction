import multiprocessing
import os

import constants
import generators
import models

name = 'dnn_bpm_class'
generator = generators.MockGenerator()
model = models.real_bpm_classification.create()

print('Train...')
model.fit_generator(
    generator=generator.get_generator(60),
    steps_per_epoch=500,
    epochs=12,
    verbose=2,
    validation_data=generator.get_generator(60),
    validation_steps=100,
    workers=multiprocessing.cpu_count(),
)

os.makedirs(constants.SAVE_PATH, exist_ok=True)
with open(f'{constants.SAVE_PATH}/{name}_model.json', "w") as json_file: json_file.write(model.to_json())
model.save_weights(f'{constants.SAVE_PATH}/{name}_weights.h5')
