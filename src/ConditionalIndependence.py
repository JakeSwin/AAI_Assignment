#############################################################################
# ConditionalIndependence.py
#
# Implements functionality for conditional independence tests via the
# library causal-learn (https://github.com/cmu-phil/causal-learn), which
# can be used to identify edges to keep or remove in a graph given a dataset.
# The flag 'chi_square_test' can be used to change tests between X^2 and G^2.
#
# This requires installing the following (at Uni-Lincoln computer labs):
# 1. Type Anaconda Prompt in your Start icon
# 2. Open your terminal as administrator
# 3. Execute=> pip install causal-learn
#
# USAGE instructions to run this program can be found at the bottom of this file.
#
# Version: 1.0, Date: 19 October 2022 (first version)
# Version: 1.1, Date: 07 October 2023 (minor revision)
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys
import csv
from causallearn.utils.cit import CIT


class ConditionalIndependence:
    chisq_obj = None
    rand_vars = []
    rv_all_values = []
    chi_square_test = True

    def __init__(self, file_name):
        data = self.read_data(file_name)
        test = "chisq" if self.chi_square_test else "gsq"
        self.chisq_obj = CIT(data, test)

    def read_data(self, data_file):
        print("\nREADING data file %s..." % data_file)
        print("---------------------------------------")

        with open(data_file) as csv_file:
            datareader = csv.reader(csv_file, delimiter=",")
            for row in datareader:
                if len(self.rand_vars) == 0:
                    self.rand_vars = row
                else:
                    self.rv_all_values.append(row)

        print("RANDOM VARIABLES=%s" % self.rand_vars)
        print("VARIABLE VALUES (first 10)=%s" % (self.rv_all_values[:10]) + "\n")
        return self.rv_all_values

    @staticmethod
    def parse_test_args(test_args: str) -> tuple[str, str, list[str]]:
        """
        :param test_args: Conditional independence test string
        :return: Variables 1, 2 and Parent strings
        """
        main_args = test_args[2:len(test_args) - 1]
        variables = main_args.split('|')[0]
        v_i = variables.split(',')[0]
        v_j = variables.split(',')[1]
        parents_i = [p for p in main_args.split('|')[1].split(',') if p.lower() != 'none']

        return v_i, v_j, parents_i

    def compute_p_value(self, variable_i, variable_j, parents_i):
        var_i = self.rand_vars.index(variable_i)
        var_j = self.rand_vars.index(variable_j)
        par_i = map(self.rand_vars.index, parents_i) if len(parents_i) != 0 else None
        p = self.chisq_obj(var_i, var_j, par_i)

        print("X2test: Vi=%s, Vj=%s, pa_i=%s, p=%s" %
              (variable_i, variable_j, parents_i, p))
        return p


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: ConditionalIndependence.py [train_file.csv] [I(Vi,Vj|parents)]")
        print(r"EXAMPLE1: python ConditionalIndependence.py lang_detect_train.csv \"I(X1,X2|Y)\“")
        print(r"EXAMPLE2: python ConditionalIndependence.py lang_detect_train.csv \"I(X1,X15|Y)\“")
        exit(0)
    else:
        data_file = sys.argv[1]
        test_args = sys.argv[2]

        ci = ConditionalIndependence(data_file)
        Vi, Vj, parents_i = ci.parse_test_args(test_args)
        ci.compute_p_value(Vi, Vj, parents_i)
