import logging
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.datasets import TUDataset


def knn_classifier(config, distances, valid_indices, n_repeat=5, test_size=0.2):
    """
    k-NN classification using a precomputed distance matrix over a subset of valid indices.

    distances: numpy array of shape [N_subset, N_subset] corresponding to valid indices.
    """
    
    # Load dataset
    dataset = TUDataset(
        root=config.dataset_dir, 
        name=config.dataset,
    )

    labels_all = dataset._data.y.numpy()

    # Select labels corresponding to valid indices
    # distances = distances[np.ix_(valid_indices, valid_indices)]
    labels = labels_all[valid_indices]
    num_classes = int(dataset.num_classes)
    
    average = 'binary' if num_classes == 2 else 'micro'
    ks = (3, 5, 7, 9, 11)
    
    all_test_acc = []
    all_test_f1 = []

    splitter = StratifiedShuffleSplit(
        n_splits=n_repeat, 
        test_size=test_size,
        random_state=42
    )

    for split_id, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        # Build distance matrices for train and test splits
        train_distance_matrix = distances[np.ix_(train_idx, train_idx)]
        np.fill_diagonal(train_distance_matrix, 1000)

        test_distance_matrix = distances[np.ix_(test_idx, train_idx)]

        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        # Find best k with CV
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        best_k = None
        best_score = 0
        
        for k in ks:
            knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed')
            scores = cross_val_score(
                knn,
                train_distance_matrix,
                train_labels,
                cv=kf
            )
            if scores.mean() > best_score:
                best_score, best_k = scores.mean(), k
        
        # Test
        knn_test = KNeighborsClassifier(n_neighbors=best_k, metric='precomputed')
        knn_test.fit(train_distance_matrix, train_labels)
        preds = knn_test.predict(test_distance_matrix)

        acc = accuracy_score(test_labels, preds)
        f1 = f1_score(test_labels, preds, average=average)

        all_test_acc.append(acc)
        all_test_f1.append(f1)

        logging.info(f"[Run {split_id+1}] best_k={best_k}, CV={best_score:.4f}, Test Acc={acc:.4f}, Test F1={f1:.4f}")
    
    logging.info(f"Final results over {n_repeat} random splits:")
    logging.info(f"Acc mean +/- std: {np.mean(all_test_acc):.4f} +/- {np.std(all_test_acc):.4f}")
    logging.info(f"F1 mean +/- std: {np.mean(all_test_f1):.4f} +/- {np.std(all_test_f1):.4f}")