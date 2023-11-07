#############################################################################
# NB_Classifier.py
#
# Implements the Naive Bayes classifier for simple probabilistic inference.
# It assumes the existence of data in CSV format, where the first line contains
# the names of random variables -- the last being the variable to predict.
# This implementation aims to be agnostic of the data (no hardcoded vars/probs)
#
# Version: 1.0, Date: 03 October 2022
# Version: 1.1, Date: 03 October 2023 more compatible with CPT_Generator
# Version: 1.2, Date: 08 October 2023 more compatible with ModelEvaluator
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys
import math
import time
import csv


class NBClassifier:
    rand_vars = []
    rv_key_values = {}
    rv_all_values = []
    predictor_variable = None
    num_data_instances = 0
    default_missing_count = 0.000001
    probabilities = {}
    predictions = []
    log_probabilities = False

    def __init__(self, file_name, fitted_model=None):
        if file_name is None:
            return
        else:
            self.read_data(file_name)

        if fitted_model is None:
            self.training_time = time.time()
            self.estimate_probabilities()
            self.training_time = time.time() - self.training_time

        else:
            self.inference_time = time.time()
            self.rv_key_values = fitted_model.rv_key_values
            self.probabilities = fitted_model.probabilities
            self.training_time = fitted_model.training_time
            self.test_learnt_probabilities(file_name)
            self.inference_time = time.time() - self.inference_time

    def read_data(self, data_file):
        print("\nREADING data file %s..." % data_file)
        print("---------------------------------------")

        self.rand_vars = []
        self.rv_key_values = {}
        self.rv_all_values = []

        with open(data_file, newline='') as csv_file:
            datareader = csv.reader(csv_file, delimiter=",")
            for row in datareader:
                if len(self.rand_vars) == 0:
                    self.rand_vars = row
                    self.rv_key_values = {k: [] for k in self.rand_vars}
                else:
                    self.rv_all_values.append(row)
                    self.update_variable_key_values(row)
                    self.num_data_instances += 1

        self.predictor_variable = self.rand_vars[len(self.rand_vars) - 1]

        print("RANDOM VARIABLES=%s" % self.rand_vars)
        print("VARIABLE KEY VALUES=%s" % self.rv_key_values)
        print("VARIABLE VALUES (first 10)=%s" % (self.rv_all_values[:10]))
        print("PREDICTOR VARIABLE=%s" % self.predictor_variable)
        print("|data instances|=%d" % self.num_data_instances)

    def update_variable_key_values(self, values):
        for i in range(0, len(self.rand_vars)):
            variable = self.rand_vars[i]
            key_values = self.rv_key_values[variable]
            value_in_focus = values[i]
            if value_in_focus not in key_values:
                self.rv_key_values[variable].append(value_in_focus)

    def estimate_probabilities(self):
        countings = self.estimate_countings()
        prior_counts = countings[self.predictor_variable]

        print("\nESTIMATING probabilities...")
        for variable, counts in countings.items():
            prob_distribution = {}
            for key, val in counts.items():
                variables = key.split('|')

                if len(variables) == 1:
                    # prior probability
                    probability = float(val / self.num_data_instances)
                else:
                    # conditional probability
                    probability = float(val / prior_counts[variables[1]])

                if self.log_probabilities is False:
                    prob_distribution[key] = probability
                else:
                    # convert probability to log probability
                    prob_distribution[key] = math.log(probability)

            self.probabilities[variable] = prob_distribution

        for variable, prob_dist in self.probabilities.items():
            prob_mass = 0
            for value, prob in prob_dist.items():
                prob_mass += prob
            print("P(%s)=>%s\tSUM=%f" % (variable, prob_dist, prob_mass))

    def estimate_countings(self):
        print("\nESTIMATING countings...")

        countings = {}
        for variable_index in range(0, len(self.rand_vars)):
            variable = self.rand_vars[variable_index]

            if variable_index == len(self.rand_vars) - 1:
                # prior counts
                countings[variable] = self.get_counts(None)
            else:
                # conditional counts
                countings[variable] = self.get_counts(variable_index)

        print("countings=" + str(countings))
        return countings

    def get_counts(self, variable_index):
        counts = {}
        predictor_index = len(self.rand_vars) - 1

        # accumulate countings
        for values in self.rv_all_values:
            if variable_index is None:
                # case: prior probability
                value = values[predictor_index]
            else:
                # case: conditional probability
                value = values[variable_index] + "|" + values[predictor_index]

            try:
                counts[value] += 1
            except KeyError:
                counts[value] = 1

        # verify countings by checking missing prior/conditional counts
        if variable_index is None:
            counts = self.check_missing_prior_counts(counts)
        else:
            counts = self.check_missing_conditional_counts(counts, variable_index)

        return counts

    def check_missing_prior_counts(self, counts):
        for var_val in self.rv_key_values[self.predictor_variable]:
            if var_val not in counts:
                print("WARNING: missing count for variable=" % var_val)
                counts[var_val] = self.default_missing_count

        return counts

    def check_missing_conditional_counts(self, counts, variable_index):
        variable = self.rand_vars[variable_index]
        for var_val in self.rv_key_values[variable]:
            for pred_val in self.rv_key_values[self.predictor_variable]:
                pair = var_val + "|" + pred_val
                if pair not in counts:
                    print("WARNING: missing count for variables=%s" % pair)
                    counts[pair] = self.default_missing_count

        return counts

    def test_learnt_probabilities(self, file_name):
        print("\nEVALUATING on " + str(file_name))

        # iterate over all instances in the test data
        for instance in self.rv_all_values:
            distribution = {}
            print("Input vector=%s" % instance)

            # iterate over all values in the predictor variable
            for predictor_value in self.rv_key_values[self.predictor_variable]:
                prob_dist = self.probabilities[self.predictor_variable]
                prob = prob_dist[predictor_value]

                # iterate over all instance values except the predictor var.
                for value_index in range(0, len(instance) - 1):
                    variable = self.rand_vars[value_index]
                    value = instance[value_index]
                    prob_dist = self.probabilities[variable]
                    cond_prob = value + "|" + predictor_value

                    if self.log_probabilities is False:
                        prob *= prob_dist[cond_prob]
                    else:
                        prob += prob_dist[cond_prob]

                distribution[predictor_value] = prob

            normalised_dist = self.get_normalised_distribution(distribution)
            self.predictions.append(normalised_dist)
            print("UN-NORMALISED DISTRIBUTION=%s" % distribution)
            print("NORMALISED DISTRIBUTION=%s" % normalised_dist)
            print("---")

    def get_normalised_distribution(self, distribution):
        normalised_dist = {}
        prob_mass = 0
        for var_val, prob in distribution.items():
            prob = math.exp(prob) if self.log_probabilities is True else prob
            prob_mass += prob

        for var_val, prob in distribution.items():
            prob = math.exp(prob) if self.log_probabilities is True else prob
            normalised_prob = prob / prob_mass
            normalised_dist[var_val] = normalised_prob

        return normalised_dist


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: NB_Classifier.py [train_file.csv] [test_file.csv]")
        exit(0)
    else:
        file_name_train = sys.argv[1]
        file_name_test = sys.argv[2]
        nb_fitted = NBClassifier(file_name_train)
        nb_tester = NBClassifier(file_name_test, nb_fitted)
