__author__ = 'ClarkWong'
import pickle
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

class proba_rf(RandomForestClassifier):
    def predict(self, X):
        return RandomForestClassifier.predict_proba(self, X)


def do_cross_validation(feature_matrix, class_labels, model, threshold, num_of_folds):
    y_pred_rf = cross_validation.cross_val_predict(model, feature_matrix, class_labels, cv=num_of_folds)

    y_pred = []
    for item in y_pred_rf:
        if item[0] >= threshold:
            y_pred.append(0)
        else:
            y_pred.append(1)

    print "P; %f" % precision_score(class_labels, y_pred)
    print "R; %f" % recall_score(class_labels, y_pred)

    return y_pred

def print_pred_error(pred, actual, list_of_pair):
    size = len(pred)
    for i in range(size):
        if pred[i] != actual[i]:
            print list_of_pair[i]

if __name__ == '__main__':
    with open('feature_matrix_train_20000.pickle', 'rb') as handle:
	    feature_matrix_train = pickle.load(handle)

    with open('classlabels_train_20000.pickle', 'rb') as handle:
        classlabels_train = pickle.load(handle)

    with open('list_of_pair_20000.pickle', 'rb') as handle:
        list_of_pair = pickle.load(handle)

    rf = proba_rf(n_estimators = 100, min_samples_split=1)

    threshold = 0.38

    pred = do_cross_validation(feature_matrix_train, classlabels_train, rf, threshold, 5)

    print_pred_error(pred, classlabels_train, list_of_pair)