from itertools import combinations

# TODO: implement a function to store ground truth as dictionary
def get_ged_labels(distance_matrix):
    num_graphs = len(distance_matrix)
    graph_pairs = combinations(range(num_graphs), r=2)
    ged_labels = {}
    for i, j in graph_pairs:
        ged_labels[(i,j)] = distance_matrix[i,j]
    return ged_labels
