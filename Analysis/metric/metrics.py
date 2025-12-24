from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import silhouette_score as sk_silhouette_score
import numpy as np
import libpysal
from esda.moran import Moran
from esda.geary import Geary

from enum import Enum

class Metrics(Enum):
    ARI = "adjusted_rand_index"
    AMI = "adjusted_mutual_info"
    SILHOUETTE = "silhouette_score"
    MORANS_I = "morans_i"
    GEARY_C = "geary_c"
    
    def calculate(self, labels_true, labels_pred):
        """Calculate the metric based on the enum value"""
        if self == Metrics.ARI:
            return ari(labels_true, labels_pred)
        elif self == Metrics.AMI:
            return adjusted_mutual_info(labels_true, labels_pred)
        elif self == Metrics.SILHOUETTE:
            return silhouette_score(labels_true, labels_pred)
        elif self == Metrics.MORANS_I:
            return morans_i(labels_true, labels_pred)
        elif self == Metrics.GEARY_C:
            return geary_c(labels_true, labels_pred)
        else:
            raise ValueError(f"Unknown metric: {self}")

def ari(labels_true, labels_pred):
    return adjusted_rand_score(labels_true, labels_pred)

def adjusted_mutual_info(labels_true, labels_pred):
    return adjusted_mutual_info_score(labels_true, labels_pred)

def silhouette_score(labels, X):
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -1  # Return -1 when silhouette score cannot be calculated
    return sk_silhouette_score(X, labels)

def morans_i(labels, coordinates):
    w = libpysal.weights.KNN.from_array(coordinates, k=8)
    
    # for each unique label, compute moran's I and take the average
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    morans_i_values = []
    for label in unique_labels:
        label_mask = (labels == label).astype(int)
        mi = Moran(label_mask, w)
        morans_i_values.append(mi.I)
    return np.mean(morans_i_values)
def geary_c(labels, coordinates):
    w = libpysal.weights.KNN.from_array(coordinates, k=8)
    
    # for each unique label, compute geary's C and take the average
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    geary_c_values = []
    for label in unique_labels:
        label_mask = (labels == label).astype(int)
        gc = Geary(label_mask, w)
        geary_c_values.append(gc.C)
    return np.mean(geary_c_values)