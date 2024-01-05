import numpy as np


def svm_process(sumlist):
    sumlist = sumlist / np.max(sumlist)
    return sumlist
