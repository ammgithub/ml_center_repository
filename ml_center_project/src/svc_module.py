"""
Created on October 31, 2017

SVM classifier (sklearn built-in)

"""
import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from time import time
import pickle

__author__ = 'amm'
__date__ = "Oct 31, 2017"
__version__ = 0.0

np.set_printoptions(linewidth=100, edgeitems='all', suppress=True,
                    precision=4)

class MySVC(object):
    """
    my Support vector classification
    """
    def __init__(self):
        """
        Constructor
        """
        
        

def plot2d(trainx, trainy, clf, meshstep=0.02):
    """
    Plot simple examples that have 2-dimensional input training samples
    (2 features)

    Parameters
    ----------
    meshstep : float, (default=0.02)
               Precision in meshgrid, smaller values result in smoother functions.

    Returns
    -------
             : Figure
    """
    if trainx.shape[1] == 2:
        xmin = trainx[:, 0].min() - 3
        xmax = trainx[:, 0].max() + 3
        ymin = trainx[:, 1].min() - 3
        ymax = trainx[:, 1].max() + 3
        xx, yy = np.meshgrid(np.arange(xmin, xmax + meshstep, meshstep),
                          np.arange(ymin, ymax + meshstep, meshstep))

        fig, ax = plt.subplots(1, 1)
        testx = np.c_[xx.ravel(), yy.ravel()]
        Z = clf.predict(testx)
        Z = Z.reshape(xx.shape)
        # colormap is coolwarm
        out = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(trainx[:, 0], trainx[:, 1], c=trainy,
                   cmap=plt.cm.coolwarm, s=60, edgecolors='k')
        ax.set_xlabel('trainx[:, 0] - Attribute 1')
        ax.set_ylabel('trainx[:, 1] - Attribute 2')
        title_string = "SVC: Training data and decision surface for: \nKernel = %s, " \
                       "degree =  %1.1f, gamma =  %1.1f, coef0 =  %1.1f" % (
                        clf.kernel, clf.degree, clf.gamma, clf.coef0)
        ax.set_title(title_string)
        plt.grid()
        plt.show()
                          
if __name__ == '__main__':
    """
    
    Examples
    --------
    >>> clf = svm.SVC(C=10000.0, cache_size=200, \
                  coef0=1, degree=2, gamma=12, kernel='rbf')
    >>> fkc = clf = svm.SVC()

    Elaborate example (OR problem)
    ------------------------------
    >>> trX = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    >>> trY = [1, -1, 1, -1]
    >>> tsX = np.array([[1, 2], [-3, 2], [6, -1]])
    >>> tsY = [1, -1, 1]
    >>> kernel = 'rbf'; degree = 2; gamma = 1; coef0 = 1
    >>> svc_C = 10000.0;svc_kernel = 'rbf';svc_degree = 2;
    >>> svc_gamma = 12; svc_coef0 = 1;svc_cache_size = 200
    >>> clf = svm.SVC(C=svc_C, cache_size=svc_cache_size, \
                  coef0=svc_coef0, degree=svc_degree, \
                  gamma=svc_gamma, kernel=svc_kernel)
    >>> print clf
    >>> clf.fit(trX, trY)
    >>> ftest = clf.predict(tsX)

    execfile('svc_module.py')
    """
    import os
    os.chdir('C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\ml_center_project\\src')

    print 80 * "-"
    print 18 * " " + "Support vector machine classification tests (Python built-in)"
    print 80 * "-"
    print 2 * " " + "(1) SVC: Testing OR problem"
    print 2 * " " + "(2) SVC: Testing AND problem"
    print 2 * " " + "(3) SVC: Testing 2-dimensional circular data"
    print 2 * " " + "(4) SVC: IRIS dataset: Testing 2-attribute, 2-class version (samples 0,...,99, classes 0, 1)"
    print 2 * " " + "(5) SVC: IRIS dataset: Testing 4-attribute, 2-class version (samples 0,...,99, classes 0, 1)"
    print 2 * " " + "(6) SVC: BREAST CANCER dataset Testing (all samples)"
    print 2 * " " + "(7) SVC: Computing generalization error for BREAST CANCER dataset (100 experiments)"
    print 80 * "-"

    bad_input = True
    while bad_input:
        try:
            user_in = int(raw_input("Make selection: "))
            bad_input = False
        except ValueError as e:
            print "%s is not a valid selection. Please try again. "\
                  % e.args[0].split(':')[1]

    print "\n\n"
    if user_in == 1:
        print "(1) SVC: Testing OR problem \n"
        # Testing OR
        trX = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        trY = [1, -1, 1, -1]
        tsX = np.array([[1, 2], [-3, 2], [6, -1]])
        tsY = [1, -1, 1]

        svc_kernel = 'rbf';svc_degree = 2;svc_gamma = 1;svc_coef0 = 1;
        svc_cache_size = 200;svc_C = 10000.0
        print "kernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f"\
              % (svc_kernel, svc_degree, svc_gamma, svc_coef0)
        print "-----------------------------------------------------"
        clf = svm.SVC(C=svc_C, cache_size=svc_cache_size, \
                      coef0=svc_coef0, degree=svc_degree, \
                      gamma=svc_gamma, kernel=svc_kernel)
        print clf
        clf.fit(trX, trY)
        print "Print something to check solution quality..."
        ftest = clf.predict(tsX)
        print "tsX = \n", tsX
        print "clf.predict(tsX) = \n", ftest
        print "tsY = \n", tsY
        if not (abs(ftest - tsY) <= 0.001).all():
            print "*** Test set not classified correctly. ***"
        ftest = clf.predict(trX)
        print "trX = \n", trX
        print "clf.predict(trX) = \n", ftest
        print "trY = \n", trY
        if not (abs(ftest - trY) <= 0.001).all():
            print "*** TRAINING SET NOT CLASSIFIED CORRECTLY. ***"
        plot2d(trX, trY, clf, 0.02)
    elif user_in == 2:
        print "(1) SVC: Testing AND problem \n"
        # Testing AND
        trX = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        trY = [1, -1, -1, -1]
        tsX = np.array([[1, 2], [-3, 2], [6, -1]])
        tsY = [1, -1, 1]

        svc_kernel = 'poly';svc_degree = 5;svc_gamma = 1;svc_coef0 = 1;
        svc_cache_size = 200;svc_C = 10000.0
        print "kernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f"\
              % (svc_kernel, svc_degree, svc_gamma, svc_coef0)
        print "-----------------------------------------------------"
        clf = svm.SVC(C=svc_C, cache_size=svc_cache_size, \
                      coef0=svc_coef0, degree=svc_degree, \
                      gamma=svc_gamma, kernel=svc_kernel)
        print clf
        clf.fit(trX, trY)
        print "Print something to check solution quality..."
        ftest = clf.predict(tsX)
        print "tsX = \n", tsX
        print "clf.predict(tsX) = \n", ftest
        print "tsY = \n", tsY
        if not (abs(ftest - tsY) <= 0.001).all():
            print "*** Test set not classified correctly. ***"
        ftest = clf.predict(trX)
        print "trX = \n", trX
        print "clf.predict(trX) = \n", ftest
        print "trY = \n", trY
        if not (abs(ftest - trY) <= 0.001).all():
            print "*** TRAINING SET NOT CLASSIFIED CORRECTLY. ***"
        plot2d(trX, trY, clf, 0.02)
    elif user_in == 7:
        print "(7) SVC: Computing generalization error for BREAST CANCER dataset (100 experiments)"
        #####################################################################
        # Testing BREAST CANCER                                             #
        # Classes:              2                                           #
        # Samples per class:    212(Malignant), 357(Benign)                 #
        # Samples total:        569                                         #
        # Dimensionality:       30                                          #
        # Features:             real, positive                              #
        #####################################################################

        myseed = 2
        np.random.seed(myseed)

        # All inputs and all labels
        bc_data = datasets.load_breast_cancer()
        scaler = MinMaxScaler()
        trx_all = bc_data.data
        trx_all = scaler.fit_transform(trx_all)
        try_all = bc_data.target
        try_all = np.array([i if i == 1 else -1 for i in try_all])
        
        # Improve performance by reshuffling all samples before train-test split
        tr_all = np.hstack((trx_all, try_all.reshape(len(try_all), 1)))
        np.random.shuffle(tr_all)
        trx_all = tr_all[:, :30]
        try_all = tr_all[:, 30]

        (num_samples, num_features) = trx_all.shape

        # Training-to-test ratio of 67% : 33% (Bennett, Mangasarian, 1992)
        train_ratio = 0.67
        test_ratio = 0.33
        num_train_samples = int(round(.67*num_samples, 0))  # 381 train
        num_test_samples = int(round(.33*num_samples, 0))  # 188 test
        assert num_train_samples + num_test_samples == num_samples, \
            "Please check the number of training and test samples. "

        t = time()
        svc_gen_error_list = []
        num_experiments = 100
        print "Running %d experiments... \n" % num_experiments
        for i in range(num_experiments):
            print "Running experiment: %d" % (i+1)
            # sorting not necessary, np.array not necessary
            tr_idx = np.random.choice(num_samples, num_train_samples,
                                             replace=False)
            ts_idx = list(set(range(num_samples)) - set(list(tr_idx)))
            trX = trx_all[tr_idx, :]
            trY = try_all[tr_idx]
            tsX = trx_all[ts_idx, :]
            tsY = try_all[ts_idx]

            kernel = 'rbf';degree = 2;gamma = 20;coef0 = 1

            svc_C = 1e9;svc_cache_size = 200

            clf = svm.SVC(C=svc_C, cache_size=svc_cache_size, coef0=coef0, \
                          degree=degree, gamma=gamma, kernel=kernel)

            clf.fit(trX, trY)
            ftest = clf.predict(tsX)

            print "Print something to check solution quality..."
            num_wrong = 1 * (ftest != tsY).sum()
            svc_gen_error_list.append(num_wrong / float(num_test_samples))

        if kernel == 'rbf':
            pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                        'ml_center_project\\ml_center_results\\'
            filename = 'svc_gen_error_%s_gamma_%d_coef_%d_seed_%d_num_exper_%d.pickle' \
                       % (kernel, gamma, coef0, myseed, num_experiments)
            f = open(pathname + filename, 'w')
            pickle.dump(svc_gen_error_list, f)
            f.close()
        elif kernel == 'poly':
            pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                        'ml_center_project\\ml_center_results\\'
            filename = 'svc_gen_error_%s_degree_%d_coef_%d_seed_%d_num_exper_%d.pickle' \
                       % (kernel, degree, coef0, myseed, num_experiments)
            f = open(pathname + filename, 'w')
            pickle.dump(svc_gen_error_list, f)
            f.close()
        else:
            # Linear kernel: degree = 1, coef0 = 0
            pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                        'ml_center_project\\ml_center_results\\'
            filename = 'svc_gen_error_%s_degree_1_coef_0_seed_%d_num_exper_%d.pickle' \
                       % (kernel, myseed, num_experiments)
            f = open(pathname + filename, 'w')
            pickle.dump(svc_gen_error_list, f)
            f.close()

        print "Generatlization error BREAST CANCER dataset (100 experiments): \n", \
            np.array(svc_gen_error_list)
        print "Elapsed time %4.1f seconds." % (time() - t)

