# coding=utf-8

from sklearn import svm
from sklearn import tree
from sklearn import metrics

import DataProcess


if __name__ == '__main__':
    data = DataProcess.bird_data_process()

    # clf = svm.SVC()
    # clf.fit(data[0], data[2])
    # pred = clf.predict(data[1])
    # print "SVM", metrics.accuracy_score(data[3], pred)

    # clf = tree.DecisionTreeClassifier()
    # clf.fit(data[0], data[2])
    # pred = clf.predict(data[1])
    # print "DecisionTree", metrics.accuracy_score(data[3], pred)

    