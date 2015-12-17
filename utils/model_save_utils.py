from __future__ import absolute_import
from __future__ import print_function

import keras
import os

DEFAULT_PATH = "../models"

def save_keras_model(name='defaultModel', model=None, include_weights=True):
    keras_models_path = os.path.dirname(DEFAULT_PATH+"/keras/")
    if not os.path.exists(keras_models_path):
        os.makedirs(keras_models_path)

    json_string = model.to_json()
    open(os.path.join(keras_models_path, name+'.json'), 'w').write(json_string)
    if (include_weights):
        model.save_weights(os.path.join(keras_models_path, name+'.h5'))

    return True

def load_keras_model(name=None, include_weights=True):
    if (name==None):
        return None

    keras_models_path = os.path.dirname(DEFAULT_PATH+"/keras/")
    model = keras.models.model_from_json(open(os.path.join(keras_models_path, name+'.json')).read())
    if (include_weights):
        model.load_weights(os.path.join(keras_models_path, name+'.h5'))
    return model