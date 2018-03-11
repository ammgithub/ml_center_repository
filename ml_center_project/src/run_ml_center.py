"""
Created on November 29, 2017

Running ml_center

FastKernelClassifier requires Gurobi in order to run the method fit_grb().  Otherwise run
fit(), with significantly reduced performance. This includes not finding the optimal
vector of weights in some instances.

"""

import numpy as np
from ml_center_module import FastKernelClassifier, print_output
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
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
    print 2 * " " + "(7) FKC: IRIS dataset: Gurobi soft fit generalization error for 2-class (100 experiments)"
    print 2 * " " + "(8) FKC: BREAST CANCER dataset: Testing (all samples)"
    print 2 * " " + "(9) FKC: BREAST CANCER dataset: Gurobi soft fit generalization error (100 experiments)"
    print 2 * " " + "(10) FKC: BREAST CANCER dataset: Gurobi hard fit generalization error (100 experiments)"
    print 2 * " " + "(11) FKC RBF: BREAST CANCER dataset: Computing accuracy (Grb soft) for many (gamma, C) (takes a while)"
    print 2 * " " + "(12) FKC POLY: BREAST CANCER dataset: Computing accuracy (Grb soft) for many (gamma, C) (takes a while)"
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
        kernel = 'poly'
        degree = 2
        gamma = 1
        coef0 = 1
        Csoft = 0.013

        print "kernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f, Csoft = %5.4f"\
              % (kernel, degree, gamma, coef0, Csoft)
        print "-----------------------------------------------------"
        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, Csoft=Csoft)

        # use: print fkc
        title_info = 'Scipy linprog soft fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

        title_info = 'Gurobi soft fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_grb(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

        title_info = 'Scipy linprog hard fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_hard(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

        title_info = 'Gurobi hard fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_grb_hard(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

    elif user_in == 2:
        print "(2) FKC: Testing AND problem \n"
        # Testing AND
        trX = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        trY = [1, -1, -1, -1]
        tsX = np.array([[1, 2], [-3, 2], [6, -1]])
        tsY = [1, -1, 1]

        kernel = 'poly'
        degree = 4
        gamma = 1
        coef0 = 1
        Csoft = 100

        print "kernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f, Csoft = %5.4f"\
              % (kernel, degree, gamma, coef0, Csoft)
        print "-----------------------------------------------------"
        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, Csoft=Csoft)

        # use: print fkc
        title_info = 'Scipy linprog soft fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

        title_info = 'Gurobi soft fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_grb(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

        title_info = 'Scipy linprog hard fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_hard(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

        title_info = 'Gurobi hard fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_grb_hard(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

    elif user_in == 3:
        print "(3) FKC: Testing 2-dimensional circular data \n"
        # Testing CIRCLE
        trX = np.array([[1, 1], [4, 1], [1, 4], [4, 4], [2, 2], [2, 3], [3, 2]])
        trY = [1, 1, 1, 1, -1, -1, -1]
        tsX = np.array([[0, 2], [3, 3], [6, 3]])
        tsY = [1, -1, 1]
        # Csoft = 0.10000 misclassifies the training data
        kernel = 'poly'
        degree = 2
        gamma = 1
        coef0 = 1
        Csoft = 1000

        print "kernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f, Csoft = %5.4f" \
              % (kernel, degree, gamma, coef0, Csoft)
        print "-----------------------------------------------------"
        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, Csoft=Csoft)

        # use: print fkc
        title_info = 'Scipy linprog soft fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

        title_info = 'Gurobi soft fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_grb(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

        title_info = 'Scipy linprog hard fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_hard(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

        title_info = 'Gurobi hard fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_grb_hard(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

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
        Csoft = 0.1000

        print "kernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f, Csoft = %5.4f" \
              % (kernel, degree, gamma, coef0, Csoft)
        print "-----------------------------------------------------"
        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, Csoft=Csoft)

        # use: print fkc
        # message: 'Optimization failed. Unable to find a feasible starting point.'
        # title_info = 'Scipy linprog soft fit:'
        # print "\n" + title_info
        # print 25 * "-"
        # fkc.fit(trX, trY)
        # print_output(fkc, tsX, tsY, title_info)

        title_info = 'Gurobi soft fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_grb(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

        # message: 'Optimization failed. Unable to find a feasible starting point.'
        # title_info = 'Scipy linprog hard fit:'
        # print "\n" + title_info
        # print 25 * "-"
        # fkc.fit_hard(trX, trY)
        # print_output(fkc, tsX, tsY, title_info)

        title_info = 'Gurobi hard fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_grb_hard(trX, trY)
        print_output(fkc, tsX, tsY, title_info)

    elif user_in == 5:
        print "(5) FKC: IRIS dataset: Testing 2 attribute, 2-class version (samples 0,...,99, classes 0, 1) \n"
        # Testing 2 attribute IRIS
        iris = datasets.load_iris()
        trX = iris.data[:100, :]
        # Classes 0 and 1 only
        trX = trX[:, [0, 1]]
        trY = iris.target[:100]
        trY = [i if i == 1 else -1 for i in trY]

        kernel = 'rbf'
        degree = 1
        gamma = 1
        coef0 = 1
        Csoft = 0.10000

        print "\nkernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f, Csoft = %5.4f"\
              % (kernel, degree, gamma, coef0, Csoft)
        print "-----------------------------------------------------"
        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, Csoft=Csoft)
        #####################################################################
        # kernel = linear, degree = 1, gamma = 1.00, coef0 = 1.00, Csoft = 10000.0000
        # fkc.fit() results in incorrect classification, fkc.fit_grb() is okay
        #####################################################################

        # use: print fkc
        title_info = 'Gurobi soft fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_grb(trX, trY)
        print "fkc.eps_opt = ", fkc.eps_opt
        print "fkc.weight_opt  (l+1-vector) = \n", fkc.weight_opt
        if title_info[-9:] == 'soft fit:':
            print "fkc.pen_opt (l-vector) = \n", fkc.pen_opt
        print "tr_accuracy = ", fkc.score_train()
        fkc.plot2d(this_title_info=title_info)

        title_info = 'Gurobi hard fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_grb_hard(trX, trY)
        print "fkc.eps_opt = ", fkc.eps_opt
        print "fkc.weight_opt  (l+1-vector) = \n", fkc.weight_opt
        if title_info[-9:] == 'soft fit:':
            print "fkc.pen_opt (l-vector) = \n", fkc.pen_opt
        print "tr_accuracy = ", fkc.score_train()
        fkc.plot2d(this_title_info=title_info)

    elif user_in == 6:
        print "(6) FKC: IRIS dataset: Testing 4 attribute, 2-class version(samples 0,...,99, classes 0, 1) \n"
        # Testing 4 attribute IRIS
        iris = datasets.load_iris()
        scaler = MinMaxScaler()
        trX = scaler.fit_transform(iris.data[:100, :])
        # trX = iris.data[:100, :]
        trY = iris.target[:100]
        trY = [i if i == 1 else -1 for i in trY]

        kernel = 'poly'
        degree = 4
        gamma = 1
        coef0 = 1
        Csoft = 10000

        print "\nkernel = %s, degree = %d, gamma = %3.2f, coef0 = %3.2f, Csoft = %5.4f"\
              % (kernel, degree, gamma, coef0, Csoft)
        print "-----------------------------------------------------"
        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, Csoft=Csoft)

        # use: print fkc
        title_info = 'Gurobi soft fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_grb(trX, trY)
        print "fkc.eps_opt = ", fkc.eps_opt
        print "fkc.weight_opt  (l+1-vector) = \n", fkc.weight_opt
        if title_info[-9:] == 'soft fit:':
            print "fkc.pen_opt (l-vector) = \n", fkc.pen_opt
        print "tr_accuracy = ", fkc.score_train()

        title_info = 'Gurobi hard fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_grb_hard(trX, trY)
        print "fkc.eps_opt = ", fkc.eps_opt
        print "fkc.weight_opt  (l+1-vector) = \n", fkc.weight_opt
        if title_info[-9:] == 'soft fit:':
            print "fkc.pen_opt (l-vector) = \n", fkc.pen_opt
        print "tr_accuracy = ", fkc.score_train()

    elif user_in == 7:
        print "(7) FKC: IRIS dataset: Gurobi soft fit generalization error for 2-class (100 experiments)"
        #####################################################################
        # Soft margin: Testing IRIS CLASSES 1 AND 2                                      #
        # Classes:              2                                           #
        # Samples per class:    50(Setosa), 50(versicolor)                 #
        # Samples total:        100                                         #
        # Dimensionality:       4                                          #
        # Features:             real, positive                              #
        #####################################################################
        myseed = 2
        np.random.seed(myseed)
        pd.set_option('expand_frame_repr', False)
        pd.set_option('display.max_rows', 400)

        # Use only iris setosa and iris versicolor
        scaler = MinMaxScaler()
        iris_data = datasets.load_iris()
        df = pd.DataFrame(scaler.fit_transform(iris_data.data[:100, :]))
        y = iris_data.target[:100]
        y = np.array([i if i == 1 else -1 for i in y])

        kernel = 'poly'
        degree = 2
        gamma = 1
        coef0 = 1
        Csoft = 10

        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma,
                                   coef0=coef0, Csoft=Csoft)
        # use: print fkc
        t = time()
        fkc_gen_error_list = []
        num_experiments = 100
        print "Running %d experiments... \n" % num_experiments
        for i in range(num_experiments):
            print "Running experiment: %d" % (i + 1)
            # 67 train and 33 test samples
            trX, tsX, trY, tsY = train_test_split(df, y, test_size=0.33)
            (num_test_samples, num_features) = tsX.as_matrix().shape

            fkc.fit_grb(trX.as_matrix(), trY)
            print "fkc.eps_opt = ", fkc.eps_opt
            # fkc.score() returns accuracy, want gen error
            fkc_gen_error_list.append(1 - fkc.score(tsX.as_matrix(), tsY))

        fkc_gen_error = np.array(fkc_gen_error_list).mean()
        if kernel == 'rbf':
            pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                       'ml_center_project\\ml_center_results\\'
            filename = 'fkc_iris12_%s_gamma_%d_coef_%d_Csoft_%4.4f_seed_%d_num_exper_%d_GRB_soft_gen_error_%0.4f.pickle' \
                       % (kernel, gamma, coef0, Csoft, myseed, num_experiments, fkc_gen_error)
            f = open(pathname + filename, 'w')
            pickle.dump(fkc_gen_error_list, f)
            f.close()
        elif kernel == 'poly':
            pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                       'ml_center_project\\ml_center_results\\'
            filename = 'fkc_iris12_%s_degree_%d_coef_%d_Csoft_%4.4f_seed_%d_num_exper_%d_GRB_soft_gen_error_%0.4f.pickle' \
                       % (kernel, degree, coef0, Csoft, myseed, num_experiments, fkc_gen_error)
            f = open(pathname + filename, 'w')
            pickle.dump(fkc_gen_error_list, f)
            f.close()
        else:
            # Linear kernel: degree = 1, coef0 = 0
            pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                       'ml_center_project\\ml_center_results\\'
            filename = 'fkc_iris12_%s_degree_1_coef_0_Csoft_%4.4f_seed_%d_num_exper_%d_GRB_soft_gen_error_%0.4f.pickle' \
                       % (kernel, Csoft, myseed, num_experiments, fkc_gen_error)
            f = open(pathname + filename, 'w')
            pickle.dump(fkc_gen_error_list, f)
            f.close()

        print "Generalization error IRIS dataset (%d experiments): " % num_experiments, "\n", \
            np.array(fkc_gen_error_list)
        print "\nAverage Generalization Error IRIS dataset (%d experiments): " \
              % num_experiments, "%.3f" % fkc_gen_error
        print "Elapsed time %4.1f seconds." % (time() - t)
    elif user_in == 8:
        print "(8) FKC: BREAST CANCER dataset: Testing (all samples) \n"
        #####################################################################
        # Soft margin: Testing BREAST CANCER                                             #
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

        kernel = 'poly'
        degree = 4
        gamma = 8
        coef0 = 1
        Csoft = 10.0

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

        # use: print fkc
        title_info = 'Gurobi soft fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_grb(trX, trY)
        print "fkc.eps_opt = ", fkc.eps_opt
        print "fkc.weight_opt  (l+1-vector) = \n", fkc.weight_opt
        if title_info[-9:] == 'soft fit:':
            print "fkc.pen_opt (l-vector) = \n", fkc.pen_opt
        print "tr_accuracy = ", fkc.score_train()

        title_info = 'Gurobi hard fit:'
        print "\n" + title_info
        print 25 * "-"
        fkc.fit_grb_hard(trX, trY)
        print "fkc.eps_opt = ", fkc.eps_opt
        print "fkc.weight_opt  (l+1-vector) = \n", fkc.weight_opt
        if title_info[-9:] == 'soft fit:':
            print "fkc.pen_opt (l-vector) = \n", fkc.pen_opt
        print "tr_accuracy = ", fkc.score_train()

    elif user_in == 9:
        print "(9) FKC: BREAST CANCER dataset: Gurobi soft fit generalization error (100 experiments)"
        #####################################################################
        # Soft margin: Testing BREAST CANCER                                             #
        # Classes:              2                                           #
        # Samples per class:    212(Malignant), 357(Benign)                 #
        # Samples total:        569                                         #
        # Dimensionality:       30                                          #
        # Features:             real, positive                              #
        #####################################################################
        myseed = 2
        np.random.seed(myseed)
        pd.set_option('expand_frame_repr', False)
        pd.set_option('display.max_rows', 400)

        # All inputs and all labels
        scaler = MinMaxScaler()
        bc_data = datasets.load_breast_cancer()
        df = pd.DataFrame(scaler.fit_transform(bc_data.data))
        y = bc_data.target
        y = np.array([i if i == 1 else -1 for i in y])

        kernel = 'rbf'
        degree = 2
        Csoft = 10000
        # kernel = 'poly'
        # degree = 4
        # Csoft = 10
        gamma = 2
        coef0 = 1

        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma,
                                   coef0=coef0, Csoft=Csoft)
        # use: print fkc
        t = time()
        fkc_gen_error_list = []
        num_experiments = 100
        print "Running %d experiments... \n" % num_experiments
        for i in range(num_experiments):
            print "Running experiment: %d" % (i+1)
            # 381 train and 188 test samples
            trX, tsX, trY, tsY = train_test_split(df, y, test_size=0.33)
            (num_test_samples, num_features) = tsX.as_matrix().shape

            fkc.fit_grb(trX.as_matrix(), trY)
            print "fkc.eps_opt = ", fkc.eps_opt
            # fkc.score() returns accuracy, want gen error
            fkc_gen_error_list.append(1 - fkc.score(tsX.as_matrix(), tsY))

        fkc_gen_error = np.array(fkc_gen_error_list).mean()
        if kernel == 'rbf':
            pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                        'ml_center_project\\ml_center_results\\'
            filename = 'fkc_bc_%s_gamma_%d_coef_%d_Csoft_%4.4f_seed_%d_num_exper_%d_GRB_soft_gen_error_%0.4f.pickle' \
                       % (kernel, gamma, coef0, Csoft, myseed, num_experiments, fkc_gen_error)
            f = open(pathname + filename, 'w')
            pickle.dump(fkc_gen_error_list, f)
            f.close()
        elif kernel == 'poly':
            pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                        'ml_center_project\\ml_center_results\\'
            filename = 'fkc_bc_%s_degree_%d_coef_%d_Csoft_%4.4f_seed_%d_num_exper_%d_GRB_soft_gen_error_%0.4f.pickle' \
                       % (kernel, degree, coef0, Csoft, myseed, num_experiments, fkc_gen_error)
            f = open(pathname + filename, 'w')
            pickle.dump(fkc_gen_error_list, f)
            f.close()
        else:
            # Linear kernel: degree = 1, coef0 = 0
            pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                        'ml_center_project\\ml_center_results\\'
            filename = 'fkc_bc_%s_degree_1_coef_0_Csoft_%4.4f_seed_%d_num_exper_%d_GRB_soft_gen_error_%0.4f.pickle' \
                       % (kernel, Csoft, myseed, num_experiments, fkc_gen_error)
            f = open(pathname + filename, 'w')
            pickle.dump(fkc_gen_error_list, f)
            f.close()

        print "Generalization error BREAST CANCER dataset (%d experiments): " % num_experiments, "\n", \
            np.array(fkc_gen_error_list)
        print "\nAverage Generalization Error BREAST CANCER dataset (%d experiments): " \
              % num_experiments, "%.3f" % fkc_gen_error
        print "Elapsed time %4.1f seconds." % (time() - t)
    elif user_in == 10:
        print "(10) FKC: BREAST CANCER dataset: Gurobi hard fit generalization error (100 experiments)"
        #####################################################################
        # Hard margin: Testing BREAST CANCER                                             #
        # Classes:              2                                           #
        # Samples per class:    212(Malignant), 357(Benign)                 #
        # Samples total:        569                                         #
        # Dimensionality:       30                                          #
        # Features:             real, positive                              #
        #####################################################################
        myseed = 2
        np.random.seed(myseed)
        pd.set_option('expand_frame_repr', False)
        pd.set_option('display.max_rows', 400)

        # All inputs and all labels
        scaler = MinMaxScaler()
        bc_data = datasets.load_breast_cancer()
        df = pd.DataFrame(scaler.fit_transform(bc_data.data))
        y = bc_data.target
        y = np.array([i if i == 1 else -1 for i in y])

        kernel = 'rbf'
        degree = 2
        # kernel = 'poly'
        # degree = 4
        gamma = 2
        coef0 = 1

        fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma,
                                   coef0=coef0)
        # use: print fkc
        t = time()
        fkc_gen_error_list = []
        num_experiments = 100
        print "Running %d experiments... \n" % num_experiments
        for i in range(num_experiments):
            print "Running experiment: %d" % (i + 1)
            # 381 train and 188 test samples
            trX, tsX, trY, tsY = train_test_split(df, y, test_size=0.33)
            (num_test_samples, num_features) = tsX.as_matrix().shape

            fkc.fit_grb_hard(trX.as_matrix(), trY)
            print "fkc.eps_opt = ", fkc.eps_opt
            # fkc.score() returns accuracy, want gen error
            fkc_gen_error_list.append(1 - fkc.score(tsX.as_matrix(), tsY))

        fkc_gen_error = np.array(fkc_gen_error_list).mean()
        if kernel == 'rbf':
            pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                       'ml_center_project\\ml_center_results\\'
            filename = 'fkc_bc_%s_gamma_%d_coef_%d_seed_%d_num_exper_%d_GRB_hard_gen_error_%0.4f.pickle' \
                       % (kernel, gamma, coef0, myseed, num_experiments, fkc_gen_error)
            f = open(pathname + filename, 'w')
            pickle.dump(fkc_gen_error_list, f)
            f.close()
        elif kernel == 'poly':
            pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                       'ml_center_project\\ml_center_results\\'
            filename = 'fkc_bc_%s_degree_%d_coef_%d_seed_%d_num_exper_%d_GRB_hard_gen_error_%0.4f.pickle' \
                       % (kernel, degree, coef0, myseed, num_experiments, fkc_gen_error)
            f = open(pathname + filename, 'w')
            pickle.dump(fkc_gen_error_list, f)
            f.close()
        else:
            # Linear kernel: degree = 1, coef0 = 0
            pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                       'ml_center_project\\ml_center_results\\'
            filename = 'fkc_bc_%s_degree_1_coef_0_seed_%d_num_exper_%d_GRB_hard_gen_error_%0.4f.pickle' \
                       % (kernel, myseed, num_experiments, fkc_gen_error)
            f = open(pathname + filename, 'w')
            pickle.dump(fkc_gen_error_list, f)
            f.close()

        print "Generalization error BREAST CANCER dataset (%d experiments): " % num_experiments, "\n", \
            np.array(fkc_gen_error_list)
        print "\nAverage Generalization Error BREAST CANCER dataset (%d experiments): " \
              % num_experiments, "%.3f" % fkc_gen_error
        print "Elapsed time %4.1f seconds." % (time() - t)
    elif user_in == 11:
        print "(11) FKC RBF: BREAST CANCER dataset: Gurobi soft fit generalization error (100 experiments)"
        #####################################################################
        # Soft margin: Testing BREAST CANCER                                             #
        # Classes:              2                                           #
        # Samples per class:    212(Malignant), 357(Benign)                 #
        # Samples total:        569                                         #
        # Dimensionality:       30                                          #
        # Features:             real, positive                              #
        #####################################################################
        myseed = 2
        np.random.seed(myseed)
        pd.set_option('expand_frame_repr', False)
        pd.set_option('display.max_rows', 400)

        # All inputs and all labels
        scaler = MinMaxScaler()
        bc_data = datasets.load_breast_cancer()
        df = pd.DataFrame(scaler.fit_transform(bc_data.data))
        y = bc_data.target
        y = np.array([i if i == 1 else -1 for i in y])

        kernel = 'rbf'
        degree = 2  # irrelevant
        coef0 = 1  # irrelevant

        Csoft_list = [1e4, 1e2, 1e0, 1e-2, 1e-4]
        gamma_list = [1/(2 * 3.**2), 1/(2 * 2.**2), 1/(2 * 1.**2),
                      1/(2 * 0.8**2), 1/(2 * 0.6**2), 1/(2 * 0.4**2)]
        degree_list = [3]
        accuracy_list = []

        for gamma in gamma_list:
            for Csoft in Csoft_list:
                fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma,
                                           coef0=coef0, Csoft=Csoft)
                # use: print fkc
                t = time()
                fkc_gen_error_list = []
                num_experiments = 100
                print "Running %d experiments... \n" % num_experiments
                for i in range(num_experiments):
                    print "Running experiment: %d" % (i+1)
                    # 381 train and 188 test samples
                    trX, tsX, trY, tsY = train_test_split(df, y, test_size=0.33)
                    (num_test_samples, num_features) = tsX.as_matrix().shape

                    fkc.fit_grb(trX.as_matrix(), trY)
                    print "fkc.eps_opt = ", fkc.eps_opt
                    # fkc.score() returns accuracy, want gen error
                    fkc_gen_error_list.append(1 - fkc.score(tsX.as_matrix(), tsY))

                fkc_gen_error = np.array(fkc_gen_error_list).mean()
                if kernel == 'rbf':
                    pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                                'ml_center_project\\ml_center_results\\'
                    filename = 'fkc_bc_%s_gamma_%d_coef_%d_Csoft_%4.4f_seed_%d_num_exper_%d_GRB_soft_gen_error_%0.4f.pickle' \
                               % (kernel, gamma, coef0, Csoft, myseed, num_experiments, fkc_gen_error)
                    f = open(pathname + filename, 'w')
                    pickle.dump(fkc_gen_error_list, f)
                    f.close()
                elif kernel == 'poly':
                    pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                                'ml_center_project\\ml_center_results\\'
                    filename = 'fkc_bc_%s_degree_%d_coef_%d_Csoft_%4.4f_seed_%d_num_exper_%d_GRB_soft_gen_error_%0.4f.pickle' \
                               % (kernel, degree, coef0, Csoft, myseed, num_experiments, fkc_gen_error)
                    f = open(pathname + filename, 'w')
                    pickle.dump(fkc_gen_error_list, f)
                    f.close()
                else:
                    # Linear kernel: degree = 1, coef0 = 0
                    pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                                'ml_center_project\\ml_center_results\\'
                    filename = 'fkc_bc_%s_degree_1_coef_0_Csoft_%4.4f_seed_%d_num_exper_%d_GRB_soft_gen_error_%0.4f.pickle' \
                               % (kernel, Csoft, myseed, num_experiments, fkc_gen_error)
                    f = open(pathname + filename, 'w')
                    pickle.dump(fkc_gen_error_list, f)
                    f.close()

                print "Generalization error BREAST CANCER dataset (%d experiments): " % num_experiments, "\n", \
                    np.array(fkc_gen_error_list)
                print "\nAverage Generalization Error BREAST CANCER dataset (%d experiments): " \
                      % num_experiments, "%.3f" % fkc_gen_error
                print "\nAverage Accuracy BREAST CANCER dataset (%d experiments): " \
                      % num_experiments, "%.5f" % (1 - fkc_gen_error)
                accuracy_list.append((1 - fkc_gen_error))
                print "Elapsed time %4.1f seconds." % (time() - t)
        accuracy_array = np.array(accuracy_list)
        accuracy_array = accuracy_array.reshape(6, 5)
        print "accuracy_array = \n", accuracy_array
    elif user_in == 12:
        print "(12) FKC POLY: BREAST CANCER dataset: Gurobi soft fit generalization error (100 experiments)"
        #####################################################################
        # Soft margin: Testing BREAST CANCER                                             #
        # Classes:              2                                           #
        # Samples per class:    212(Malignant), 357(Benign)                 #
        # Samples total:        569                                         #
        # Dimensionality:       30                                          #
        # Features:             real, positive                              #
        #####################################################################
        myseed = 2
        np.random.seed(myseed)
        pd.set_option('expand_frame_repr', False)
        pd.set_option('display.max_rows', 400)

        # All inputs and all labels
        scaler = MinMaxScaler()
        bc_data = datasets.load_breast_cancer()
        df = pd.DataFrame(scaler.fit_transform(bc_data.data))
        y = bc_data.target
        y = np.array([i if i == 1 else -1 for i in y])

        kernel = 'poly'
        gamma = 1
        coef0 = 1

        Csoft_list = [1e4, 1e2, 1e0, 1e-2, 1e-4]
        degree_list = [1, 2, 3, 4, 5]
        degree_list = [3]
        accuracy_list = []

        for degree in degree_list:
            for Csoft in Csoft_list:
                fkc = FastKernelClassifier(kernel=kernel, degree=degree, gamma=gamma,
                                           coef0=coef0, Csoft=Csoft)
                # use: print fkc
                t = time()
                fkc_gen_error_list = []
                num_experiments = 100
                print "Running %d experiments... \n" % num_experiments
                for i in range(num_experiments):
                    print "Running experiment: %d" % (i + 1)
                    # 381 train and 188 test samples
                    trX, tsX, trY, tsY = train_test_split(df, y, test_size=0.33)
                    (num_test_samples, num_features) = tsX.as_matrix().shape

                    fkc.fit_grb(trX.as_matrix(), trY)
                    print "fkc.eps_opt = ", fkc.eps_opt
                    # fkc.score() returns accuracy, want gen error
                    fkc_gen_error_list.append(1 - fkc.score(tsX.as_matrix(), tsY))

                fkc_gen_error = np.array(fkc_gen_error_list).mean()
                if kernel == 'rbf':
                    pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                               'ml_center_project\\ml_center_results\\'
                    filename = 'fkc_bc_%s_gamma_%d_coef_%d_Csoft_%4.4f_seed_%d_num_exper_%d_GRB_soft_gen_error_%0.4f.pickle' \
                               % (kernel, gamma, coef0, Csoft, myseed, num_experiments, fkc_gen_error)
                    f = open(pathname + filename, 'w')
                    pickle.dump(fkc_gen_error_list, f)
                    f.close()
                elif kernel == 'poly':
                    pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                               'ml_center_project\\ml_center_results\\'
                    filename = 'fkc_bc_%s_degree_%d_coef_%d_Csoft_%4.4f_seed_%d_num_exper_%d_GRB_soft_gen_error_%0.4f.pickle' \
                               % (kernel, degree, coef0, Csoft, myseed, num_experiments, fkc_gen_error)
                    f = open(pathname + filename, 'w')
                    pickle.dump(fkc_gen_error_list, f)
                    f.close()
                else:
                    # Linear kernel: degree = 1, coef0 = 0
                    pathname = 'C:\\Users\\amalysch\\PycharmProjects\\ml_center_repository\\' \
                               'ml_center_project\\ml_center_results\\'
                    filename = 'fkc_bc_%s_degree_1_coef_0_Csoft_%4.4f_seed_%d_num_exper_%d_GRB_soft_gen_error_%0.4f.pickle' \
                               % (kernel, Csoft, myseed, num_experiments, fkc_gen_error)
                    f = open(pathname + filename, 'w')
                    pickle.dump(fkc_gen_error_list, f)
                    f.close()

                print "Generalization error BREAST CANCER dataset (%d experiments): " % num_experiments, "\n", \
                    np.array(fkc_gen_error_list)
                print "\nAverage Generalization Error BREAST CANCER dataset (%d experiments): " \
                      % num_experiments, "%.3f" % fkc_gen_error
                print "\nAverage Accuracy BREAST CANCER dataset (%d experiments): " \
                      % num_experiments, "%.5f" % (1 - fkc_gen_error)
                accuracy_list.append((1 - fkc_gen_error))
                print "Elapsed time %4.1f seconds." % (time() - t)
        accuracy_array = np.array(accuracy_list)
        accuracy_array = accuracy_array.reshape(5, 5)
        print "accuracy_array = \n", accuracy_array
    else:
        print "Invalid selection. Program terminating. "
