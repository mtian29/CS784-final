__author__ = 'ClarkWong'
import pickle
import do_cv

def insert_new_feature(fn, list_of_pair_p, feature_matrix):
    add = fn(list_of_pair_p)
    for i in range(len(feature_matrix)):
        example = feature_matrix[i]
        example.append(add[i])

def remove_feature(pos, feature_matrix):
    for example in feature_matrix:
        del example[pos]

def feature_num(feature_matrix):
    return len(feature_matrix[0])

def print_out_examples(feature_matrix, start, end):
    for i in range(start, end):
        print feature_matrix[i]

###### below this line ##########
#Define the functions to generate the new feature, one function for each feature;
#After defining the functions, pass it to the parameter of insert_new_feature

##This function is just for example
def add_fn(list_of_pair_p):
    add = []
    for pair in list_of_pair_p:
        product1 = pair[0]
        product2 = pair[1]
        label = pair[2]
        something = 1
        add.append(something)
    return add


if __name__ == '__main__':
    with open('feature_matrix_train_20000.pickle', 'rb') as handle:
	    feature_matrix_train = pickle.load(handle)

    with open('classlabels_train_20000.pickle', 'rb') as handle:
        classlabels_train = pickle.load(handle)

    with open('list_of_pair_20000.pickle', 'rb') as handle:
        list_of_pair = pickle.load(handle)

    insert_new_feature(add_fn, list_of_pair, feature_matrix_train)

    print_out_examples(feature_matrix_train, 0, 10)

    remove_feature(feature_num(feature_matrix_train)-1, feature_matrix_train)

    print_out_examples(feature_matrix_train, 0, 10)

    rf = do_cv.proba_rf(n_estimators = 100, min_samples_split=1)

    threshold = 0.38

    pred = do_cv.do_cross_validation(feature_matrix_train, classlabels_train, rf, threshold, 5)

    do_cv.print_pred_error(pred, classlabels_train, list_of_pair)

    # with open('feature_matrix_train_20000.pickle', 'wb') as handle:
    #     pickle.dump(feature_matrix_train, handle)