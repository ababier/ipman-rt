import keras.backend as k_backend
import tensorflow as tf
from keras.src.engine.keras_tensor import KerasTensor
from open_kbp import DataShapes
from tensorflow import boolean_mask


def log_predicted_value_loss(y_true, y_pred):
    return tf.map_fn(lambda x: k_backend.log(k_backend.clip(x, k_backend.epsilon(), 1.0)), y_pred)


def sum_log_predicted_value_loss(y_true, y_pred):
    return tf.map_fn(lambda x: k_backend.sum(k_backend.log(k_backend.clip(x, k_backend.epsilon(), 1.0))), y_pred)


def minimize_value(y_true, y_pred):
    return y_pred


def dose_from_mask(dose, structure_mask):
    flat_dose = k_backend.reshape(dose, (-1, 1))
    flat_mask = k_backend.reshape(structure_mask, (-1, 1))
    masked_dose = boolean_mask(flat_dose, flat_mask)
    return masked_dose


def calculate_ipm_objective(dose, roi_masks, data_shapes: DataShapes):
    num_oars = len(data_shapes.rois["oars"])
    all_oar_masks = roi_masks[:, :, :, :num_oars]
    oars_mask = k_backend.greater(k_backend.sum(all_oar_masks, axis=-1), 0)
    oars_dose = dose_from_mask(dose, oars_mask)
    oar_mean_dose = k_backend.mean(oars_dose)
    objective_value = k_backend.reshape(oar_mean_dose, (1,))
    return objective_value


def weighted_sum(vector: KerasTensor, weights: tf.Tensor):
    vector_2d = k_backend.reshape(vector, (1, -1))
    weights_2d = k_backend.reshape(weights, (-1, 1))
    return k_backend.dot(vector_2d, weights_2d)
