#model3.py
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('energy_produced.h5')

def predict(features):
    prediction = model.predict(np.array(features).reshape(1, -1))
    return prediction[0][0]