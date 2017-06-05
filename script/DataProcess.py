# coding=utf-8

import scipy
import scipy.io
import sklearn


def bird_data_process():
    bird_train = scipy.io.loadmat('../data/bird_train.mat')
    bird_train = bird_train['bird_train']
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for figure in bird_train:
        # print len(figure)
        x_train.append(figure[0:4095])
        y_train.append(figure[4096])

    bird_test = scipy.io.loadmat('../data/bird_validation_new.mat')
    bird_test = bird_test['bird_validation']
    for figure in bird_test:
        # print len(figure)
        x_test.append(figure[0:4095])
        y_test.append(figure[4096])

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    bird_data_process()
