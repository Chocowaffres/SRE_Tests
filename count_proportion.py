import arff, numpy as np
from scipy import sparse

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

if __name__ == '__main__':
    dataset_name = 'dataset_7options'
    label_count = 16

    X_train, Y_train, X_test, Y_test = load_custom_dataset(dataset_name, label_count)

    labels = Y_train.shape[1]
    list_classes = []
    for i in range(labels):
        tot_eq_1 = sum(1 if y != 0 else 0 for y in Y_train[:,i])
        tot = Y_train[:,i].shape[0]
        tot_eq_0 = tot - tot_eq_1
        print("Class ", i, "  Weight Balance for 0: ", (tot / (2 * tot_eq_0)), "  Weight Balance for 1: ", (tot / (2 * tot_eq_1)))

    # Formula to calculate weight balance: wj=n_samples / (n_classes * n_samplesj)

