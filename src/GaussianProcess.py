#############################################################################
# ModelEvaluator.py
#
# Implements functionality for Gaussian Process classification via utilities
# in file gaussian_processes_util.py written by Martin Krasser
#
# This program considers a baseline classifier with two variants:
# (1) The one discussed during lecture of week 5, see slide 36
# (2) Another one based on prob. densities for 1 and 0 derived from
#     the estimated mean vector and covariance matrices of a GPR.
#     pdf_1 = self.get_gaussian_probability_density(1, self.mu[i], var[i])
#     pdf_0 = self.get_gaussian_probability_density(0, self.mu[i], var[i])
#     prob = pdf_1 / (pdf_1 + pdf_0)
# Each of this variants can be activated by setting the flag baseline_variant1
#
# Version: 1.0, Date: 23 October 2023, functionality for binary classification
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys
import time

import george
import torch
import gpytorch
import numpy as np
import GPy
from george import kernels
from scipy.optimize import minimize
from gaussian_processes_util import plot_gp
from gaussian_processes_util import nll_fn
from gaussian_processes_util import posterior
from sklearn import metrics
from ModelEvaluator import ModelEvaluator


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianProcess:
    noise = 0.4
    mu = None  # mean vector to be estimated
    cov = None  # covariance matrix to be estimated
    predictions = []  # probabilities of test data points
    running_time = None  # execution time for training+test
    baseline_variant1 = False  # use baseline discussed in lecture

    def __init__(self, implementation, datafile_train, datafile_test):
        # Load training and test data from two separate CVS files
        X_train, Y_train = self.load_csv_file(datafile_train)
        X_test, Y_test = self.load_csv_file(datafile_test)

        # train GP model via regression and evaluate it with test data
        self.running_time = time.time()
        if implementation == "GPy":
            self.m = GPy.models.GPRegression(X_train, Y_train)
            self.mu = self.m.posterior.mean
            self.cov = self.m.posterior.covariance
        if implementation == "george":
            # TODO fix this
            kernel = np.var(Y_train) * kernels.ExpSquaredKernel(0.5)
            self.gp = george.GP(kernel)
            self.gp.compute(X_train, Y_train)
        if implementation == "gpytorch":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            tensor_train_x = torch.tensor(X_train, dtype=torch.float32, device=self.device)
            tensor_train_y = torch.tensor(Y_train, dtype=torch.int, device=self.device).reshape(-1)
            self.m = ExactGPModel(
                tensor_train_x,
                tensor_train_y,
                likelihood=self.likelihood)
            self.m.train()
            self.likelihood.train()
            optimizer = torch.optim.Adam(self.m.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.m)
            for i in range(50):
                optimizer.zero_grad()
                out = self.m(tensor_train_x)
                losses = -mll(out, tensor_train_y)
                loss = losses.sum()
                loss.backward()
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, 50, loss.item(),
                    self.m.covar_module.base_kernel.lengthscale.item(),
                    self.m.likelihood.noise.item()
                ))
                optimizer.step()
            self.m.eval()
            self.likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                tensor_test_x = torch.tensor(X_test, dtype=torch.float32, device=self.device)
                self.f_preds = self.m(tensor_test_x)
            self.mu = self.f_preds.mean.numpy()
            self.cov = self.f_preds.covariance_matrix.numpy()
        else:
            self.estimate_mean_and_covariance(X_train, Y_train, X_test)
        self.running_time = time.time() - self.running_time
        self.evaluate_model_baseline(X_test, Y_test)

    def load_csv_file(self, file_name):
        print("LOADING CSV file %s" % file_name)
        X = []
        Y = []
        start_reading = False
        with open(file_name) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not start_reading:
                    # ignore the header/first line
                    start_reading = True
                else:
                    values = [float(i) for i in line.split(',')]
                    X.append(values[:len(values)-1])
                    Y.append(values[len(values)-1:])
        return np.asarray(X), np.asarray(Y)

    def estimate_mean_and_covariance(self, X_train, Y_train, X_test):
        print("ESTIMATING mean and covariance for GP model...")

        # search for optimal values for l and sigma_f via negative log-likelihood
        # and the Limited-memory BGGS-B algorithm. The latter is an extension of
        # the L-BFGS algorithm used for hyperparameter optimisation
        res = minimize(nll_fn(X_train, Y_train, self.noise, False), [1, 1], 
                       bounds=((1e-5, None), (1e-5, None)), method='L-BFGS-B')
        l_opt, sigma_f_opt = res.x

        print("Hyperparameters: l=%s sigma=%s noise=%s" % (l_opt, sigma_f_opt, self.noise))

        # Compute posterior mean and covariance using optimised kernel parameters
        self.mu, self.cov = posterior(X_test, X_train, Y_train, \
                            l=l_opt, sigma_f=sigma_f_opt, sigma_y=self.noise)

    @staticmethod
    def get_gaussian_probability_density(x, mean, var):
        e_val = -np.power((x-mean), 2)/(2*var)
        return (1/(np.sqrt(2*np.pi*var))) * np.exp(e_val)

    def evaluate_model_baseline(self, X_test, Y_test):
        mu = self.mu.reshape(-1, 1)
        var = 1.96 * np.sqrt(np.diag(self.cov)).reshape(-1, 1)
        _min = np.min(mu)
        _max = np.max(mu)
        Y_true = Y_test
        Y_pred = []
        Y_prob = []

        for i in range(0, len(X_test)):
            # This block of code calculates probabilities with two variants.
            # variant1: uses predicted means only and ignores variance info.
            # variant2: uses both predicted means and predicted variances.
            # Note that the predicted means and variances are derived from 
            # the code above, mainly from estimate_mean_and_covariance()
            if self.baseline_variant1: 
                prob = float((mu[i]-_min)/(_max-_min))
            else:
                pdf_1 = self.get_gaussian_probability_density(1, self.mu[i], var[i])
                pdf_0 = self.get_gaussian_probability_density(0, self.mu[i], var[i])
                prob = pdf_1 / (pdf_1 + pdf_0)
            Y_prob.append(prob)

            print("[%s] X_test=%s Y_test=%s mu=%s var=%s p=%s" %
                  (i, X_test[i], Y_test[i], self.mu[i], var[i], prob))
            if prob >= 0.5:
                Y_pred.append(1)
            else:
                Y_pred.append(0)

        ModelEvaluator.compute_performance(Y_true, Y_pred, Y_prob)
        print("Running Time="+str(self.running_time)+" secs.")


if len(sys.argv) != 4:
    print("USAGE: GaussignProcess.py [GP Implementation] [training_file.csv] [test_file.csv]")
    print("EXAMPLE> GaussignProcess.py GPy data_banknote_authentication-train.csv data_banknote_authentication-test.csv")
    exit(0)
else:
    implementation = sys.argv[1]
    datafile_train = sys.argv[2]
    datafile_test = sys.argv[3]
    GaussianProcess(implementation, datafile_train, datafile_test)
