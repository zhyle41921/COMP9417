import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score

def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def ROC_AUC(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)