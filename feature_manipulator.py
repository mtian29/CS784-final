__author__ = 'ClarkWong'
import pickle
import do_cv
from py_stringmatching import simfunctions, tokenizers

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
def add_fn1(list_of_pair_p):
    add = []
    for pair in list_of_pair_p:
        product1 = pair[0]
        #print ''.join(product1)
        product2 = pair[1]
        label = pair[2]
        jaccard_json = simfunctions.jaccard(tokenizers.delimiter(''.join(product1)), tokenizers.delimiter(''.join(product2)))
        add.append(jaccard_json)
    return add

def add_fn2(list_of_pair_p):
    add = []
    for pair in list_of_pair_p:
        product1 = pair[0]
        #print ''.join(product1)
        product2 = pair[1]
        label = pair[2]
        jaccard_json = simfunctions.jaccard(tokenizers.qgram(''.join(product1),3), tokenizers.qgram(''.join(product2),3))
        add.append(jaccard_json)
    return add

def add_fn4(list_of_pair_p):
    add = []
    for pair in list_of_pair_p:
        product1 = pair[0]
        product2 = pair[1]
        label = pair[2]

        if ("Manufacturer Part Number" in product1 and "Product Name" in product2):
            manu_part_number_set = tokenizers.delimiter(product1["Manufacturer Part Number"][0])
            des_set = product2["Product Name"][0]
            count = 0
            for manu_part in manu_part_number_set:
                if manu_part in des_set:
                    count = count+1
            #manu_part1_in_name2 = count/len(manu_part_number_set)
            manu_part1_in_name2 = count
        else:
            manu_part1_in_name2 = 0
        add.append(manu_part1_in_name2)
    return add

def add_fn5(list_of_pair_p):
    add = []
    for pair in list_of_pair_p:
        attribute_id1 = pair[0]
        attribute_id2 = pair[1]
        label = pair[2]

        if ("Brand" in attribute_id1 and "Product Name" in attribute_id2):
            brand_set = tokenizers.delimiter(attribute_id1["Brand"][0])
            des = attribute_id2["Product Name"][0]
            count = 0
            for brand in brand_set:
                if brand in des:
                    count = count+1
                    brand1_in_short2 = count/len(brand_set)
                else:
                    brand1_in_short2 = 0
        add.append(brand1_in_short2)
    return add


if __name__ == '__main__':
    with open('feature_matrix_train_20000.pickle', 'rU') as handle:
	    feature_matrix_train = pickle.load(handle)

    with open('classlabels_train_20000.pickle', 'rb') as handle:
        classlabels_train = pickle.load(handle)

    with open('list_of_pair_20000.pickle', 'rU') as handle:
        list_of_pair = pickle.load(handle)

    #insert_new_feature(add_fn5, list_of_pair, feature_matrix_train)

    print_out_examples(feature_matrix_train, 0, 10)

    #remove_feature(feature_num(feature_matrix_train)-1, feature_matrix_train)

    print_out_examples(feature_matrix_train, 0, 10)

    import sys
    sys.stdout = open("out.txt","w")

    rf = do_cv.proba_rf(n_estimators = 100, min_samples_split=1)

    threshold = 0.47

    pred = do_cv.do_cross_validation(feature_matrix_train, classlabels_train, rf, threshold, 5)
    do_cv.print_pred_error(pred, classlabels_train, list_of_pair)



    with open('feature_matrix_train_20000.pickle', 'wb') as handle:
         pickle.dump(feature_matrix_train, handle)

