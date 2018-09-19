from typing import Optional

from keras import Model
from keras.engine.saving import model_from_json

import constants
import os


def load_model(name) -> Optional[Model]:
    model_file = f'{constants.SAVE_PATH}/{name}_model.json'
    weights_file = f'{constants.SAVE_PATH}/{name}_weights.h5'
    if not os.path.exists(model_file) or not os.path.exists(weights_file): return None

    with open(model_file, "r") as f: model_json = f.read()

    model = model_from_json(model_json)
    model.load_weights(weights_file)

    return model
