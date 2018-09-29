from typing import Optional

from keras import Model
from keras.engine.saving import model_from_json

import constants
import os


def load_model(name, weights=None) -> Optional[Model]:
    if weights is None: weights = f'{name}_weights.h5'

    model_file = f'{constants.SAVE_PATH}/{name}.json'
    weights_file = f'{constants.SAVE_PATH}/{weights}.h5'
    if not os.path.exists(model_file) or not os.path.exists(weights_file): return None

    with open(model_file, "r") as f: model_json = f.read()

    model = model_from_json(model_json)
    model.load_weights(weights_file)

    return model
