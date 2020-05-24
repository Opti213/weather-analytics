
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets, metrics
from sklearn.metrics import roc_curve, auc




def get_false_roc_dots():
    y = np.array([0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35])
    fpr, tpr, _ = metrics.roc_curve(y, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def render_roc(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2

    plt.plot(
        fpr,
        tpr,
        color='darkorange',
        lw=lw,
        label='ROC curve (area = %0.2f)' % roc_auc
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig("tmp1.png")

a,b,c = get_false_roc_dots()
render_roc(a,b,c)