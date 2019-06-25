import tensorflow as tf
import keras
import keras.backend as K
from keras.models import model_from_json
import json
import numpy as np

from . import GridSearch_Consts as GS_Util

def build_model(training_params, model_weights_filename, model_json_filename):


    optimizer = get_optimizer(training_params['optimizer_params'])

    model_loss = get_loss(training_params['model_params'][GS_Util.LOSS_FN()])

    model_params = training_params[GS_Util.MODEL_PARAMS()]

    json_file = open(model_json_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # now load weights

    model.load_weights(model_weights_filename)
    # set the compilation parameters

    model.compile(optimizer=optimizer,
                  loss=model_loss,
                  metrics=[dsc, concurrency])

    return model


def get_loss(loss_name, **kwargs):

    loss_fn_dict = {'dsc': dice_loss,
                    'dsc_bce': dice_cross_entropy_loss,
                    'bce': keras.losses.binary_crossentropy}

    assert loss_name in loss_fn_dict.keys(), 'Must select a loss function that exists.'


    return loss_fn_dict[loss_name]


def eval_metrics(metric_name):

    metric_dict = {'dice': dice,
                   'concurrency': concurrency,
                   'accuracy': tf.metrics.accuracy}


    return metric_dict[metric_name]


def get_optimizer(optimizer_params):

    opt_name = optimizer_params[GS_Util.OPTIMIZER()]

    opt_dict = {'adam': keras.optimizers.adam,
                'Adam': keras.optimizers.adam,
                'SGD': keras.optimizers.SGD}

    opt_fn = opt_dict[opt_name]

    optimizer_params.pop(GS_Util.OPTIMIZER())

    optimizer = opt_fn(**optimizer_params)

    return optimizer

def dsc(y_true, y_pred, **kwargs):
    """This method calculates the dice coefficient between the true
     and predicted masks
    Args:
        y_true: The true mask(i.e. ground-truth or expert annotated mask)
        y_pred: The predicted mask

    Returns:
        double: The dice coefficient"""

    smooth = K.epsilon()
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    # return (2. * intersection + smooth) / (
    #         K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_loss(y_true, y_pred):
    """This method calculates the loss function based on dice coeff
    Args:
        y_true: The true mask(i.e. ground truth or expert annotated mask)
        y_pred: The predicted mask

    Returns:
        double: The dice coefficient based loss function
    """
    return -dsc(y_true, y_pred)


def dice_cross_entropy_loss(y_true, y_pred):
    """This method calculates the loss function based on dice coeff
    Args:
        y_true: The true mask(i.e. ground truth or expert annotated mask)
        y_pred: The predicted mask

    Returns:
        double: The dice coefficient based loss function
    """
    a = 0.3

    return ((1 - dsc(y_true, y_pred)) + a * keras.losses.binary_crossentropy(y_true, y_pred)) / (1 + a)



def concurrency(y_true, y_pred):
    prediction_threshold = 0.5
    y_pred = K.cast(K.greater(y_pred, prediction_threshold), K.floatx())


    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)


    intersection = K.sum(y_true_f * y_pred_f)

    return ((intersection / K.sum(y_true_f)) + (intersection / K.sum(y_pred_f))) / 2.

