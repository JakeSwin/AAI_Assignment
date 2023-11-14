import numpy as np
import sys
from BayesNetUtil import tokenise_query

def get_discretized_vector(X, query):
    _mean = np.mean(X)
    _std = np.std(X)
    bins = [_mean-(_std*2), _mean-(_std*1), _mean, _mean+(_std*1), _mean+(_std*2)]
    X_discretized = np.digitize(X, bins)
    Q = np.digitize(query, bins)
    return X_discretized, Q

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: DiscretizeQuery.py [full_data.csv] [query]")
        print("EXAMPLE1> DiscretizeQuery.py diabetes_data-discretized.csv \"P(B|J=true,M=true)\"")
        exit(0)

    file_name = sys.argv[1]  # inference algorithm={InferenceByEnumeration,RejectionSampling}
    prob_query = sys.argv[2]  # your_config_file.txt, e.g., config-alarm.txt
    prob_query = tokenise_query(prob_query)
    with open(file_name) as f:
        first_line = f.readline().strip().split(',')
    data = np.loadtxt(file_name, skiprows=1, delimiter=",")
    fake_data = []
    for var in first_line:
        if var in prob_query["evidence"].keys():
            fake_data.append(prob_query["evidence"][var])
        else:
            fake_data.append(0)
    discretized, query = get_discretized_vector(data, fake_data)
    print(query)
    print("Done")
