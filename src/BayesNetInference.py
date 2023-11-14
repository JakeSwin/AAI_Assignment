#############################################################################
# BayesNetInference.py
#
# This program implements the algorithm "Inference by Enumeration", which
# makes use of BayesNetsReader to facilitate reading data of a Bayes net via
# the object self.bn created by the inherited class (BayesNetReader). It also
# makes use of miscellaneous methods implemented in BayesNetUtil.
# Its purpose is to answer probabilistic queries such as P(Y|X=true,Z=false).
# This implementation is agnostic of the data and provides a general
# implementation that can not used across datasets by providing a config file.
#
# This program also implements the algorithm "Rejection Sampling", which
# imports functionalities to facilitate reading data of a Bayes net via
# the object self.bn created by the inherited class BayesNetReader.
# Its purpose is to answer probabilistic queries such as P(Y|X=true,Z=false).
# This implementation is agnostic of the data and provides a general
# implementation that can be used across datasets by providing a config file.
#
# WARNINGS:
#    (1) This code has been revised but not thoroughly tested.
#    (2) The execution time depends on the number of random samples.
#
# Version: 1.0, Date: 06 October 2022, 1st version of class BayesNetExactInference
# Version: 1.1, Date: 07 October 2022, 1st version of class BayesNetApproxInference
# Version: 1.2, Date: 21 October 2022, revised code to be more query compatible
# Version: 1.3, Date: 20 October 2022, revised code made simpler (more intuitive)
# Version: 1.4, Date: 13 October 2023, merged classes for unified prob. inference
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys
import random
from numpy.random import choice
import time
from itertools import chain
from BayesNetUtil import (get_domain_values, get_parents, get_children, tokenise_query, get_probability_given_parents,
                          normalise, get_probability_given_markov_blanket)
from BayesNetReader import BayesNetReader


class BayesNetInference(BayesNetReader):
    query = {}
    prob_dist = {}

    def __init__(self, alg_name, file_name, prob_query, num_samples):
        super().__init__(file_name)
        if alg_name is None or prob_query is None or num_samples is None:
            return
        self.query = tokenise_query(prob_query)

        start = time.time()
        if alg_name == 'InferenceByEnumeration':
            self.prob_dist = self.enumeration_ask()
            normalised_dist = normalise(self.prob_dist)
            print("un-normalised P(%s)=%s" % (self.query["query_var"], self.prob_dist))
            print("normalised P(%s)=%s" % (self.query["query_var"], normalised_dist))

        elif alg_name == 'RejectionSampling':
            self.prob_dist = self.rejection_sampling(num_samples)
            print("P(%s)=%s" % (self.query["query_var"], self.prob_dist))

        elif alg_name == 'LikelihoodWeighting':
            self.prob_dist = self.likelihood_weighting(num_samples)
            print("P(%s)=%s" % (self.query["query_var"], self.prob_dist))

        elif alg_name == 'GibbsSampling':
            self.prob_dist = self.gibbs_sampling(num_samples)
            print("P(%s)=%s" % (self.query["query_var"], self.prob_dist))

        else:
            print("ERROR: Couldn't recognise algorithm="+str(alg_name))
            print("Valid choices={InferenceByEnumeration,RejectionSampling}")

        end = time.time()
        print('Execution Time: {}'.format(end-start))

    # main method for inference by enumeration, which invokes
    # enumerate_all() for each domain value of the query variable
    def enumeration_ask(self):
        print("\nSTARTING Inference by Enumeration...")
        Q = {}
        for value in self.bn["rv_key_values"][self.query["query_var"]]:
            value = value.split('|')[0]
            Q[value] = 0

        for value, probability in Q.items():
            value = value.split('|')[0]
            variables = [x.split("P(")[1].split(")")[0].split("|")[0] for x in self.bn["structure"]]
            evidence = self.query["evidence"].copy()
            evidence[self.query["query_var"]] = value
            probability = self.enumerate_all(variables, evidence)
            Q[value] = probability

        print("\tQ="+str(Q))
        return Q  # Q is an un-normalised probability distribution

    # returns a probability for the arguments provided, based on
    # summations or multiplications of prior/conditional probabilities
    def enumerate_all(self, variables, evidence):
        if len(variables) == 0:
            return 1.0

        V = variables[0]

        if V in evidence:
            v = evidence[V].split('|')[0]
            p = get_probability_given_parents(V, v, evidence, self.bn)
            variables.pop(0)
            return p*self.enumerate_all(variables, evidence)

        else:
            total = 0
            evidence_copy = evidence.copy()
            for v in get_domain_values(V, self.bn):
                evidence[V] = v
                p = get_probability_given_parents(V, v, evidence, self.bn)
                rest_variables = variables.copy()
                rest_variables.pop(0)
                total += p*self.enumerate_all(rest_variables, evidence)
                evidence = evidence_copy

            return total

    # main method to carry out approximate probabilistic inference
    # which invokes prior_sample() and is_compatible_with_evidence()
    def rejection_sampling(self, num_samples):
        print("\nSTARTING rejection sampling...")
        query_variable = self.query["query_var"]
        evidence = self.query["evidence"]
        samples = []  # vector of non-rejected samples
        C = {}  # counts per value in query variable

        # initialise vector of counts
        for value in self.bn["rv_key_values"][query_variable]:
            value = value.split("|")[0]
            C[value] = 0

        # loop to increase counts when the sampled vector X consistent w/evidence
        for i in range(0, num_samples):
            X = self.prior_sample(evidence)
            if X is not None and self.is_compatible_with_evidence(X, evidence):
                value_to_increase = X[query_variable]
                C[value_to_increase] += 1

        try:
            print("Countings of query_variable %s=%s" % (query_variable, C))
            return normalise(C)
        except:
            print("ABORTED due to insufficient number of samples...")
            exit(0)

    def gibbs_sampling(self, num_samples):
        query_variable = self.query["query_var"]
        evidence = self.query["evidence"]
        non_evidence_vars = [i for i in self.bn["random_variables"] if i not in evidence.keys()]
        C = {}  # counts per value in query variable

        # initialise vector of counts
        for value in self.bn["rv_key_values"][query_variable]:
            value = value.split("|")[0]
            C[value] = 0

        # Generate initial sample state
        sample = evidence.copy()
        for var in non_evidence_vars:
            sample[var] = random.choice(get_domain_values(var, self.bn))

        for i in range(0, num_samples):
            for var in non_evidence_vars:
                dist = get_probability_given_markov_blanket(var, sample, self.bn)
                sample[var] = choice(list(dist.keys()), p=list(dist.values()))
                C[sample[query_variable]] += 1

        return normalise(C)

    def likelihood_weighting(self, num_samples):
        query_variable = self.query["query_var"]
        evidence = self.query["evidence"]
        C = {}  # counts per value in query variable

        # initialise vector of counts
        for value in self.bn["rv_key_values"][query_variable]:
            value = value.split("|")[0]
            C[value] = []

        for i in range(0, num_samples):
            event = evidence.copy()
            weight = 1
            target = None
            for var in [x.split("P(")[1].split(")")[0].split("|")[0] for x in self.bn["structure"]]:
                if var in event:
                    p = get_probability_given_parents(var, event[var], event, self.bn)
                    weight = weight * p
                    result = event[var]
                else:
                    result = self.get_sampled_value(var, event)
                    event[var] = result
                if var == query_variable:
                    target = result
            C[target].append(weight)

        return {k: (sum(C[k]) / sum(chain.from_iterable(C.values()))) for k in C.keys()}

    # returns a dictionary of sampled values for each of the random variables
    def prior_sample(self, evidence):
        X = {}
        sampled_var_values = {}

        # iterates over the set of random variables as specified in the
        # config file of the Bayes net, in the order from left to right
        for variable in [x.split("P(")[1].split(")")[0].split("|")[0] for x in self.bn["structure"]]:
            X[variable] = self.get_sampled_value(variable, sampled_var_values)
            sampled_var_values[variable] = X[variable]
            if variable in evidence and evidence[variable] != X[variable]:
                return None

        return X

    # returns a sampled value for the given random variable as argument
    def get_sampled_value(self, V, sampled):
        parents = get_parents(V, self.bn)
        cumulative_cpt = {}
        prob_mass = 0

        # generate cumulative distribution for random variable V without parents
        if parents is None:
            for value, probability in self.bn["CPT("+V+")"].items():
                prob_mass += probability
                cumulative_cpt[value] = prob_mass

        # generate cumulative distribution for random variable V with parents
        else:
            for v in get_domain_values(V, self.bn):
                p = get_probability_given_parents(V, v, sampled, self.bn)
                prob_mass += p
                cumulative_cpt[v] = prob_mass

        # check that the probabilities sum to 1 (or almost)
        if prob_mass < 0.999 or prob_mass > 1:
            print("ERROR: probabilities=%s do not sum to 1" % (cumulative_cpt))
            exit(0)

        # sample a value from the cumulative distribution generated above
        for value, probability in cumulative_cpt.items():
            random_number = random.random()
            if random_number <= probability:
                return value.split("|")[0]

        return None  # this shouldn't happen -- unless something was incompatible

    # returns True if evidence has key-value pairs same as X
    # returns False otherwise
    @staticmethod
    def is_compatible_with_evidence(X, evidence):
        compatible = True
        for variable, value in evidence.items():
            if X[variable] != value:
                compatible = False
                break
        return compatible


if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("USAGE: BayesNetInference.py [inference_algorithm] [your_config_file.txt] [query] (num_samples)")
        print("EXAMPLE1> BayesNetInference.py InferenceByEnumeration config-alarm.txt \"P(B|J=true,M=true)\"")
        print("EXAMPLE2> BayesNetInference.py RejectionSampling config-alarm.txt \"P(B|J=true,M=true)\" 10000")
        exit(0)

    alg_name = sys.argv[1]  # inference algorithm={InferenceByEnumeration,RejectionSampling}
    file_name = sys.argv[2]  # your_config_file.txt, e.g., config-alarm.txt
    prob_query = sys.argv[3]  # query, e.g., P(B|J=true,M=true)
    num_samples = int(sys.argv[4]) if len(sys.argv) == 5 else None  # number of samples, e.g., 10000

    BayesNetInference(alg_name, file_name, prob_query, num_samples)
