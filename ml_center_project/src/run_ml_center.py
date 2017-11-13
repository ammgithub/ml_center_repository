"""
Created on October 25, 2017

Running ml_center

FastKernelClassifier requires Gurobi in order to run the method fit_grb().  Otherwise run
fit(), with significantly reduced performance. This includes not finding the optimal
vector of weights in some instances.

"""

import numpy as np
from ml_center_module import FastKernelClassifier
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from time import time
import pickle

__author__ = 'amm'
__date__ = "Oct 25, 2017"
__version__ = 0.0

np.set_printoptions(precision=4, edgeitems=None, linewidth=100, suppress=True)

if __name__ == '__main__':
    """
    execfile('run_ml_center.py')
    """
    import os
    os.chdir('C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\ml_center_project\\src')

    print 80 * "-"
    print 18 * " " + "Machine learning center tests (2-class separation)"
    print 80 * "-"
    print 2 * " " + "(1) FKC: Testing OR problem"
    print 2 * " " + "(2) FKC: Testing AND problem"
    print 2 * " " + "(3) FKC: Testing 2-dimensional circular data"
    print 2 * " " + "(4) FKC: Testing extended 2-dimensional circular data"
    print 2 * " " + "(5) FKC: IRIS dataset: Testing 2-attribute, 2-class version (samples 0,...,99, classes 0, 1)"
    print 2 * " " + "(6) FKC: IRIS dataset: Testing 4-attribute, 2-class version (samples 0,...,99, classes 0, 1)"
    print 2 * " " + "(7) FKC: Computing generalization error for 2-class IRIS dataset(100 experiments)"
    print 2 * " " + "(8) FKC: BREAST CANCER dataset Testing (all samples)"
    print 2 * " " + "(9) FKC: Computing generalization error for BREAST CANCER dataset (100 experiments)"
    print 80 * "-"

    user_in = 0
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
        print "(1) FKC: Testing OR problem \n"
        # Testing OR
        trX = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        trY = [1, -1, 1, -1]
        tsX = np.array([[1, 2], [-3, 2], [6, -1]])
        tsY = [1, -1, 1]

        # kernel = 'poly'; degree = 1; gamma = 1; coef0 = 1; Csoft = 0.010  # solution
        # kernel = 'poly'; degree = 1; gamma = 1; coef0 = 1; Csoft = 0.012  # solution
        # kernel = 'poly'; degree = 1; gamma = 1; coef0 = 1; Csoft = 0.013  # no solution
        # kernel = 'poly'; degree = 1; gamma = 1; coef0 = 1; Csoft = 0.015  # solution
        # kernel = 'poly'; degree = 1; gamma = 1; coef0 = 1; Csoft = 0.017  # solution
        kernel = 'poly'; degree = 2; gamma = 1; coef0 = 1; Csoft = 100
        print "kernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f, Csoft = %5.4f"\
              % (kernel, degree, gamma, coef0, Csoft)
        print "-----------------------------------------------------"
        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, Csoft=Csoft)
        fkc.fit_grb(trX, trY)
        # fkc.fit(trX, trY)
        print "fkc.eps_opt = ", fkc.eps_opt
        print "fkc.weight_opt  (l+1-vector) = \n", fkc.weight_opt
        print "fkc.pen_opt (l-vector) = \n", fkc.pen_opt
        print "fkc.fun_opt = ", fkc.fun_opt
        ftest = fkc.predict(tsX)
        print "tsX = \n", tsX
        print "fkc.predict(tsX) = \n", ftest
        print "tsY = \n", tsY
        if not (abs(ftest - tsY) <= 0.001).all():
            print "*** Test set not classified correctly. ***"
        ftest = fkc.predict(trX)
        print "trX = \n", trX
        print "fkc.predict(trX) = \n", ftest
        print "trY = \n", trY
        if not (abs(ftest - trY) <= 0.001).all():
            print "*** TRAINING SET NOT CLASSIFIED CORRECTLY. ***"
        fkc.plot2d(0.02)
    elif user_in == 2:
        print "(2) FKC: Testing AND problem \n"
        # Testing AND
        trX = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        trY = [1, -1, -1, -1]
        tsX = np.array([[1, 2], [-3, 2], [6, -1]])
        tsY = [1, -1, 1]

        # kernel = 'poly'; degree = 1; gamma = 1; coef0 = 1; Csoft = 0.10000  # shift in separator
        # kernel = 'poly'; degree = 1; gamma = 1; coef0 = 1; Csoft = 0.00001  # incomplete sep
        kernel = 'poly'; degree = 1; gamma = 1; coef0 = 1; Csoft = 100
        print "kernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f, Csoft = %5.4f"\
              % (kernel, degree, gamma, coef0, Csoft)
        print "-----------------------------------------------------"
        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, Csoft=Csoft)
        fkc.fit_grb(trX, trY)
        # fkc.fit(trX, trY)
        print "fkc.eps_opt = ", fkc.eps_opt
        print "fkc.weight_opt  (l+1-vector) = \n", fkc.weight_opt
        print "fkc.pen_opt (l-vector) = \n", fkc.pen_opt
        print "fkc.fun_opt = ", fkc.fun_opt
        ftest = fkc.predict(tsX)
        print "tsX = \n", tsX
        print "fkc.predict(tsX) = \n", ftest
        print "tsY = \n", tsY
        if not (abs(ftest - tsY) <= 0.001).all():
            print "*** Test set not classified correctly. ***"
        ftest = fkc.predict(trX)
        print "trX = \n", trX
        print "fkc.predict(trX) = \n", ftest
        print "trY = \n", trY
        if not (abs(ftest - trY) <= 0.001).all():
            print "*** TRAINING SET NOT CLASSIFIED CORRECTLY. ***"
        fkc.plot2d(0.02)
    elif user_in == 3:
        print "(3) FKC: Testing 2-dimensional circular data \n"
        # Testing CIRCLE
        trX = np.array([[1, 1], [4, 1], [1, 4], [4, 4], [2, 2], [2, 3], [3, 2]])
        trY = [1, 1, 1, 1, -1, -1, -1]
        tsX = np.array([[0, 2], [3, 3], [6, 3]])
        tsY = [1, -1, 1]
        # kernel = 'poly'; degree = 2; gamma = 1; coef0 = 1; Csoft = 0.10000  # one point misclass
        kernel = 'poly'; degree = 2; gamma = 1; coef0 = 1; Csoft = 1000

        print "kernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f, Csoft = %5.4f" \
              % (kernel, degree, gamma, coef0, Csoft)
        print "-----------------------------------------------------"
        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, Csoft=Csoft)
        fkc.fit_grb(trX, trY)
        # fkc.fit(trX, trY)
        print "fkc.eps_opt = ", fkc.eps_opt
        print "fkc.weight_opt  (l+1-vector) = \n", fkc.weight_opt
        print "fkc.pen_opt (l-vector) = \n", fkc.pen_opt
        print "fkc.fun_opt = ", fkc.fun_opt
        ftest = fkc.predict(tsX)
        print "tsX = \n", tsX
        print "fkc.predict(tsX) = \n", ftest
        print "tsY = \n", tsY
        if not (abs(ftest - tsY) <= 0.001).all():
            print "*** Test set not classified correctly. ***"
        ftest = fkc.predict(trX)
        print "trX = \n", trX
        print "fkc.predict(trX) = \n", ftest
        print "trY = \n", trY
        if not (abs(ftest - trY) <= 0.001).all():
            print "*** TRAINING SET NOT CLASSIFIED CORRECTLY. ***"
        fkc.plot2d(0.02)
    elif user_in == 4:
        print "(4) FKC: Testing extended 2-dimensional circular data \n"
        # Testing extended CIRCLE
        trX = np.array([[1, 1], [4, 1], [1, 4], [4, 4], [2, 2], [2, 3], [3, 2], [5, 4.5]])
        trY = [1, 1, 1, 1, -1, -1, -1, -1]
        tsX = np.array([[0, 2], [3, 3], [6, 3]])
        tsY = [1, -1, 1]
        kernel = 'rbf'
        degree = 2
        gamma = 0.1
        coef0 = 1
        Csoft = 1000.

        print "kernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f, Csoft = %5.4f" \
              % (kernel, degree, gamma, coef0, Csoft)
        print "-----------------------------------------------------"
        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, Csoft=Csoft)
        fkc.fit_grb(trX, trY)
        # fkc.fit(trX, trY)
        print "fkc.eps_opt = ", fkc.eps_opt
        print "fkc.weight_opt  (l+1-vector) = \n", fkc.weight_opt
        print "fkc.pen_opt (l-vector) = \n", fkc.pen_opt
        print "fkc.fun_opt = ", fkc.fun_opt
        ftest = fkc.predict(tsX)
        print "tsX = \n", tsX
        print "fkc.predict(tsX) = \n", ftest
        print "tsY = \n", tsY
        if not (abs(ftest - tsY) <= 0.001).all():
            print "*** Test set not classified correctly. ***"
        ftest = fkc.predict(trX)
        print "trX = \n", trX
        print "fkc.predict(trX) = \n", ftest
        print "trY = \n", trY
        if not (abs(ftest - trY) <= 0.001).all():
            print "*** TRAINING SET NOT CLASSIFIED CORRECTLY. ***"
        fkc.plot2d(0.02)
    elif user_in == 5:
        print "(5) FKC: Testing 2 attribute, 2-class version of IRIS dataset (samples 0,...,99, classes 0, 1) \n"
        # Testing 2 attribute IRIS
        iris = datasets.load_iris()
        trX = iris.data[:100, :]
        # Classes 0 and 1 only
        trX = trX[:, [0, 1]]
        trY = iris.target[:100]
        trY = [i if i == 1 else -1 for i in trY]
        kernel = 'linear'; degree = 1; gamma = 1; coef0 = 1; Csoft = 10000

        print "\nkernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f, Csoft = %5.4f"\
              % (kernel, degree, gamma, coef0, Csoft)
        print "-----------------------------------------------------"
        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, Csoft=Csoft)
        #####################################################################
        # kernel = linear, degree = 1, gamma = 1.00, coef0 = 1.00, Csoft = 10000.0000
        # fkc.fit() results in incorrect classification, fkc.fit_grb() is okay
        #####################################################################
        fkc.fit_grb(trX, trY)
        # fkc.fit(trX, trY)
        print "fkc.eps_opt = ", fkc.eps_opt
        print "fkc.weight_opt  (l+1-vector) = \n", fkc.weight_opt
        print "fkc.pen_opt (l-vector) = \n", fkc.pen_opt
        print "fkc.fun_opt = ", fkc.fun_opt
        ftest = fkc.predict(trX)
        print "trX = ", trX
        print "fkc.predict(trX) = ", ftest
        print "trY = ", trY
        if not (abs(ftest - trY) <= 0.001).all():
            print "*** Training set not classified correctly. ***"
        print "No prodiction done."
        fkc.plot2d(0.02)
    elif user_in == 6:
        print "(6) FKC: Testing 4 attribute, 2-class version of IRIS dataset (samples 0,...,99, classes 0, 1) \n"
        # Testing 4 attribute IRIS
        iris = datasets.load_iris()
        scaler = MinMaxScaler()
        trX = scaler.fit_transform(iris.data[:100, :])
        # trX = iris.data[:100, :]
        trY = iris.target[:100]
        trY = [i if i == 1 else -1 for i in trY]
        kernel = 'linear'; degree = 1; gamma = 1; coef0 = 1; Csoft = 10000

        print "\nkernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f, Csoft = %5.4f"\
              % (kernel, degree, gamma, coef0, Csoft)
        print "-----------------------------------------------------"
        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, Csoft=Csoft)
        fkc.fit_grb(trX, trY)
        # fkc.fit(trX, trY)
        print "fkc.eps_opt = ", fkc.eps_opt
        print "fkc.weight_opt  (l+1-vector) = \n", fkc.weight_opt
        print "fkc.pen_opt (l-vector) = \n", fkc.pen_opt
        print "fkc.fun_opt = ", fkc.fun_opt
        ftest = fkc.predict(trX)
        print "trX = ", trX
        print "fkc.predict(trX) = ", ftest
        print "trY = ", np.array(trY, 'float')
        if not (abs(ftest - trY) <= 0.001).all():
            print "*** Training set not classified correctly. ***"
        print "No prodiction done."
        fkc.plot2d(0.02)
    elif user_in == 7:
        print "(7) FKC: Computing generalization error for 2-class IRIS dataset (100 experiments)"
        #####################################################################
        # Testing IRIS CLASSES 1 AND 2                                      #
        # Classes:              2                                           #
        # Samples per class:    50(Setosa), 50(versicolor)                 #
        # Samples total:        100                                         #
        # Dimensionality:       4                                          #
        # Features:             real, positive                              #
        #####################################################################
        myseed = 2
        np.random.seed(myseed)

        # All inputs and all labels
        iris_data = datasets.load_iris()
        scaler = MinMaxScaler()
        trx_all = iris_data.data[:100, :]
        trx_all = scaler.fit_transform(trx_all)
        try_all = iris_data.target[:100]
        try_all = np.array([i if i == 1 else -1 for i in try_all])

        # Improve performance by reshuffling all samples before train-test split
        tr_all = np.hstack((trx_all, try_all.reshape(len(try_all), 1)))
        np.random.shuffle(tr_all)
        trx_all = tr_all[:, :4]
        try_all = tr_all[:, 4]

        (num_samples, num_features) = trx_all.shape

        # Training-to-test ratio of 67% : 33% (Bennett, Mangasarian, 1992)
        train_ratio = 0.67
        test_ratio = 0.33
        num_train_samples = int(round(.67 * num_samples, 0))  # 67 train
        num_test_samples = int(round(.33 * num_samples, 0))  # 33 test
        assert num_train_samples + num_test_samples == num_samples, \
            "Please check the number of training and test samples. "

        kernel = 'poly'
        degree = 2
        gamma = 1
        coef0 = 1
        Csoft = 10

        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma,
                                   coef0=coef0, Csoft=Csoft)

        t = time()
        fkc_gen_error_list = []
        num_experiments = 100
        print "Running %d experiments... \n" % num_experiments
        for i in range(num_experiments):
            print "Running experiment: %d" % (i + 1)
            # sorting not necessary, np.array not necessary
            tr_idx = np.random.choice(num_samples, num_train_samples,
                                      replace=False)
            ts_idx = list(set(range(num_samples)) - set(list(tr_idx)))
            trX = trx_all[tr_idx, :]
            trY = try_all[tr_idx]
            tsX = trx_all[ts_idx, :]
            tsY = try_all[ts_idx]

            fkc.fit_grb(trX, trY)
            # fkc.fit(trX, trY)
            ftest = fkc.predict(tsX)
            print "fkc.eps_opt = ", fkc.eps_opt
            # print "fkc.weight_opt  (l+1-vector) = \n", fkc.weight_opt
            # print "fkc.pen_opt (l-vector) = \n", fkc.pen_opt
            # print "fkc.fun_opt = ", fkc.fun_opt
            num_wrong = 1 * (ftest != tsY).sum()
            fkc_gen_error_list.append(num_wrong / float(num_test_samples))

        if kernel == 'rbf':
            pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                       'ml_center_project\\ml_center_results\\'
            filename = 'fkc_iris12_gen_error_%s_gamma_%d_coef_%d_Csoft_%4.4f_seed_%d_num_exper_%d_GRB.pickle' \
                       % (kernel, gamma, coef0, Csoft, myseed, num_experiments)
            f = open(pathname + filename, 'w')
            pickle.dump(fkc_gen_error_list, f)
            f.close()
        elif kernel == 'poly':
            pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                       'ml_center_project\\ml_center_results\\'
            filename = 'fkc_iris12_gen_error_%s_degree_%d_coef_%d_Csoft_%4.4f_seed_%d_num_exper_%d.pickle' \
                       % (kernel, degree, coef0, Csoft, myseed, num_experiments)
            f = open(pathname + filename, 'w')
            pickle.dump(fkc_gen_error_list, f)
            f.close()
        else:
            # Linear kernel: degree = 1, coef0 = 0
            pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                       'ml_center_project\\ml_center_results\\'
            filename = 'fkc_iris12_gen_error_%s_degree_1_coef_0_Csoft_%4.4f_seed_%d_num_exper_%d.pickle' \
                       % (kernel, Csoft, myseed, num_experiments)
            f = open(pathname + filename, 'w')
            pickle.dump(fkc_gen_error_list, f)
            f.close()

        print "Generatlization error BREAST CANCER dataset (%d experiments): " % num_experiments, "\n", \
            np.array(fkc_gen_error_list)
        print "\nAverage Generatlization Error BREAST CANCER dataset (%d experiments): " \
              % num_experiments, "%.3f" % np.array(fkc_gen_error_list).mean()
        print "Elapsed time %4.1f seconds." % (time() - t)
    elif user_in == 8:
        print "(8) FKC: Testing BREAST CANCER dataset (all samples) \n"
        #####################################################################
        # Testing BREAST CANCER                                             #
        # Classes:              2                                           #
        # Samples per class:    212(Malignant), 357(Benign)                 #
        # Samples total:        569                                         #
        # Dimensionality:       30                                          #
        # Features:             real, positive                              #
        #####################################################################
        bc_data = datasets.load_breast_cancer()
        scaler = MinMaxScaler()
        trX = bc_data.data
        trX = scaler.fit_transform(trX)
        trY = bc_data.target
        trY = np.array([i if i == 1 else -1 for i in trY])
        kernel = 'rbf'
        degree = 1
        gamma = 8
        coef0 = 1
        Csoft = 10000

        print "\nkernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f, Csoft = %5.4f"\
              % (kernel, degree, gamma, coef0, Csoft)
        print "-----------------------------------------------------"
        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, Csoft=Csoft)
        #####################################################################
        # kernel = rbf, degree = 1, gamma = 8.00, coef0 = 1.00, Csoft = 10000.0000
        # Here, the Scipy optimizer does not identify an optimal solution.
        # (all weights are zero, fkc.fun_opt = -0)
        # Gurobi finds optimal solution,
        # (nonzero weights, fkc.fun_opt (GRB) =  -0.45512124496)
        #####################################################################
        fkc.fit_grb(trX, trY)
        # fkc.fit(trX, trY)
        print "fkc.eps_opt = ", fkc.eps_opt
        print "fkc.weight_opt  (l+1-vector) = \n", fkc.weight_opt
        print "fkc.pen_opt (l-vector) = \n", fkc.pen_opt
        print "fkc.fun_opt = ", fkc.fun_opt
        ftest = fkc.predict(trX)
        print "Skipped printing trX ...\n "
        print "fkc.predict(trX) = \n", ftest
        print "trY = \n", trY
        if not (abs(ftest - trY) <= 0.001).all():
            print "*** TRAINING SET NOT CLASSIFIED CORRECTLY. ***"
        print "No prodiction done."
        fkc.plot2d(0.02)
    elif user_in == 9:
        print "(9) FKC: Computing generalization error for BREAST CANCER dataset (100 experiments)"
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

        kernel = 'rbf'
        degree = 2
        gamma = 12
        coef0 = 1
        Csoft = 10000

        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma,
                                   coef0=coef0, Csoft=Csoft)

        t = time()
        fkc_gen_error_list = []
        num_experiments = 100
        print "Running %d experiments... \n" % num_experiments
        for i in range(num_experiments):
            print "Running experiment: %d" % (i+1)
            # 381 train and 188 test samples
            train_set, test_set = train_test_split(tr_all, test_size=0.33)
            trX = train_set[:, :30]
            trY = train_set[:, 30]
            tsX = test_set[:, :30]
            tsY = test_set[:, 30]
            (num_test_samples, num_features) = tsX.shape

            fkc.fit_grb(trX, trY)
            # fkc.fit(trX, trY)
            ftest = fkc.predict(tsX)
            print "fkc.eps_opt = ", fkc.eps_opt
            # print "fkc.weight_opt  (l+1-vector) = \n", fkc.weight_opt
            # print "fkc.pen_opt (l-vector) = \n", fkc.pen_opt
            # print "fkc.fun_opt = ", fkc.fun_opt
            num_wrong = 1 * (ftest != tsY).sum()
            fkc_gen_error_list.append(num_wrong / float(num_test_samples))

        if kernel == 'rbf':
            pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                        'ml_center_project\\ml_center_results\\'
            filename = 'fkc_bc_gen_error_%s_gamma_%d_coef_%d_Csoft_%4.4f_seed_%d_num_exper_%d_GRB.pickle' \
                       % (kernel, gamma, coef0, Csoft, myseed, num_experiments)
            f = open(pathname + filename, 'w')
            pickle.dump(fkc_gen_error_list, f)
            f.close()
        elif kernel == 'poly':
            pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                        'ml_center_project\\ml_center_results\\'
            filename = 'fkc_bc_gen_error_%s_degree_%d_coef_%d_Csoft_%4.4f_seed_%d_num_exper_%d.pickle' \
                       % (kernel, degree, coef0, Csoft, myseed, num_experiments)
            f = open(pathname + filename, 'w')
            pickle.dump(fkc_gen_error_list, f)
            f.close()
        else:
            # Linear kernel: degree = 1, coef0 = 0
            pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                        'ml_center_project\\ml_center_results\\'
            filename = 'fkc_bc_gen_error_%s_degree_1_coef_0_Csoft_%4.4f_seed_%d_num_exper_%d.pickle' \
                       % (kernel, Csoft, myseed, num_experiments)
            f = open(pathname + filename, 'w')
            pickle.dump(fkc_gen_error_list, f)
            f.close()

        print "Generatlization error BREAST CANCER dataset (%d experiments): " % num_experiments, "\n", \
            np.array(fkc_gen_error_list)
        print "\nAverage Generatlization Error BREAST CANCER dataset (%d experiments): " \
              % num_experiments, "%.3f" % np.array(fkc_gen_error_list).mean()
        print "Elapsed time %4.1f seconds." % (time() - t)
    else:
        print "Invalid selection. Program terminating. "
    print "Finished."




