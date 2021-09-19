### https://xang1234.github.io/multi-label/
import sklearn.metrics as metrics
from skmultilearn.dataset import load_dataset, save_to_arff

from skmultilearn.problem_transform import ClassifierChain, BinaryRelevance, LabelPowerset
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from skmultilearn.adapt import BRkNNaClassifier, MLkNN
from skmultilearn.ensemble import LabelSpacePartitioningClassifier, MajorityVotingClassifier
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder, NetworkXLabelGraphClusterer, FixedLabelSpaceClusterer
from sklearn.multiclass import OneVsRestClassifier

import arff, numpy as np
from scipy import sparse

import cProfile

# --------------------------- Datasets -----------------------------

def load_yeast_dataset():
    X_train, Y_train, _, _ = load_dataset("yeast", "train")
    X_test, Y_test, _, _ = load_dataset("yeast", "test")

    return X_train, Y_train, X_test, Y_test

def load_custom_dataset(dataset_name, label_count, length_train=None, length_test=None):
    train_dataset = arff.load(open(dataset_name+'_train.arff', 'r'))

    if length_train == None:
        length_train = len(train_dataset['data'])

    test_dataset = arff.load(open(dataset_name+'_test.arff', 'r'))

    if length_test == None:
        length_test = len(test_dataset['data'])

    X_train = np.array([np.array(train_dataset['data'][i], dtype=float)[:-label_count] for i in range(length_train)])
    Y_train = np.array([np.array(train_dataset['data'][i], dtype=int)[-label_count:] for i in range(length_train)])

    X_test = np.array([np.array(test_dataset['data'][i], dtype=float)[:-label_count] for i in range(length_test)])
    Y_test = np.array([np.array(test_dataset['data'][i], dtype=int)[-label_count:] for i in range(length_test)])

    X_train = sparse.lil_matrix(X_train, shape=X_train.shape)
    Y_train = sparse.lil_matrix(Y_train, shape=Y_train.shape)
    X_test = sparse.lil_matrix(X_test, shape=X_test.shape)
    Y_test = sparse.lil_matrix(Y_test, shape=Y_test.shape)

    return X_train, Y_train, X_test, Y_test



# ---------------------------- Classifiers ---------------------------

def predict_classifier_chain(base_classifier, X_train, Y_train, X_test):
    # Classifier Chains
    cc = ClassifierChain(
        classifier=base_classifier, 
        require_dense=[False, True],
    )
    return cc.fit(X_train, Y_train).predict(X_test)

def predict_binary_relevance(base_classifier, X_train, Y_train, X_test):
    # Binary Relevance
    br = BinaryRelevance(
        classifier=base_classifier,
        require_dense=[False, True],
    )
    return br.fit(X_train, Y_train).predict(X_test)

def predict_label_powerset(base_classifier, X_train, Y_train, X_test):
    # Label Powerset
    lp = LabelPowerset(
        classifier=base_classifier,
        require_dense=[False, True],
    )
    return lp.fit(X_train, Y_train).predict(X_test)

def predict_multilabel_k_nearest_neighbors(X_train, Y_train, X_test):
    # Multi-label k-Nearest Neighbors
    mlknn = MLkNN(
        k=1,
        s=0.5,
    )
    return mlknn.fit(X_train, Y_train).predict(X_test)

def predict_binary_relevance_k_nearest_neighbors(X_train, Y_train, X_test):
    # Binary Relevance k-Nearest Neighbors
    brknn = BRkNNaClassifier(
        k=3,    
    )
    return brknn.fit(X_train, Y_train).predict(X_test)

def predict_label_space_partitioning_classifier(base_classifier, X_train, Y_train, X_test):
    graph_builder = LabelCooccurrenceGraphBuilder(weighted=True,
                                                  include_self_edges=False)
    clusterer = NetworkXLabelGraphClusterer(graph_builder, method='louvain')

    lspc = LabelSpacePartitioningClassifier(
        classifier=BinaryRelevance(
            classifier=base_classifier,
            require_dense=[False, True],
        ),
        clusterer=clusterer
    )
    return lspc.fit(X_train, Y_train).predict(X_test)

def predict_majority_voting_classifier(base_classifier, X_train, Y_train, X_test):
    graph_builder = LabelCooccurrenceGraphBuilder(weighted=True,
                                                  include_self_edges=False)
    clusterer = NetworkXLabelGraphClusterer(graph_builder, method='louvain')

    mvc = MajorityVotingClassifier(
        classifier=BinaryRelevance(
            classifier=base_classifier,
            require_dense=[False, True],
    ),
        clusterer=clusterer
    )
    return mvc.fit(X_train, Y_train).predict(X_test)

# ---------------------------- Metrics ---------------------------

def calculate_metrics(Y_hat, Y_test):
    accuracy = metrics.accuracy_score(Y_test, Y_hat)
    f1_score = metrics.f1_score(Y_test, Y_hat, average='weighted')

    return accuracy, f1_score



# ----------------------------- Main ------------------------------

def main():
    dataset_name = 'datasets/dataset_normal'
    label_count = 16

    # length_train diz a quantidade de dados para a parte de treino do modelo, se a variável não for definida então utiliza o conjunto de treino todo
    # length_test diz a quantidade de dados para a parte de teste do modelo, se a variável não for definida então utiliza o conjunto de teste todo
    X_train, Y_train, X_test, Y_test = load_custom_dataset(dataset_name, label_count)
    # X_train, Y_train, X_test, Y_test = load_yeast_dataset()

    classifier = DecisionTreeClassifier()

    cc_Y_hat = predict_classifier_chain(classifier, X_train, Y_train, X_test)
    cc_ac, cc_f1 = calculate_metrics(cc_Y_hat, Y_test)

    print("############# Classifier Chains ############")
    print("Accuracy: ", cc_ac, "F1-micro: ", cc_f1)

    br_Y_hat = predict_binary_relevance(classifier, X_train, Y_train, X_test)
    br_ac, br_f1 = calculate_metrics(br_Y_hat, Y_test)

    print("############# Binary Relevance ############")
    print("Accuracy: ", br_ac, "F1-micro: ", br_f1)

    lp_Y_hat = predict_label_powerset(classifier, X_train, Y_train, X_test)
    lp_ac, lp_f1 = calculate_metrics(lp_Y_hat, Y_test)

    print("############# Label Powerset ############")
    print("Accuracy: ", lp_ac, "F1-micro: ", lp_f1)

    # mlknn_Y_hat = predict_multilabel_k_nearest_neighbors(X_train, Y_train, X_test)
    # mlknn_ac, mlknn_f1 = calculate_metrics(mlknn_Y_hat, Y_test)

    # print("############# Multi-label k-Nearest Neighbors ############")
    # print("Accuracy: ", mlknn_ac, "F1-micro: ", mlknn_f1)

    # brknn_Y_hat = predict_binary_relevance_k_nearest_neighbors(X_train, Y_train, X_test)
    # brknn_ac, brknn_f1 = calculate_metrics(brknn_Y_hat, Y_test)

    # print("############# Binary Relevance k-Nearest Neighbors ############")
    # print("Accuracy: ", brknn_ac, "F1-micro: ", brknn_f1)

    lspc_Y_hat = predict_label_space_partitioning_classifier(classifier, X_train, Y_train, X_test)
    lspc_ac, lspc_f1 = calculate_metrics(lspc_Y_hat, Y_test)

    print("############# Label Space Partitioning Classifier ############")
    print("Accuracy: ", lspc_ac, "F1-micro: ", lspc_f1)

    mvc_Y_hat = predict_majority_voting_classifier(classifier, X_train, Y_train, X_test)
    mvc_ac, mvc_f1 = calculate_metrics(mvc_Y_hat, Y_test)

    print("############# Majority Voting Classifier ############")
    print("Accuracy: ", mvc_ac, "F1-micro: ", mvc_f1)

if __name__ == '__main__':
    cProfile.run('main()')