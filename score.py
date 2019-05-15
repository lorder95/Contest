import numpy as np


def score(y_true, y_pred):
    corr = np.sum(y_true == y_pred)
    ast = np.sum(y_pred == -1)
    wrong = np.sum(y_true != y_pred)-ast

    #return round(((corr-wrong) / y_pred.shape[0])*100,2)

    return (corr/(corr+wrong+ast)*100)
