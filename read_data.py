__author__ = 'ClarkWong'
import json
import pickle

def generate_list_of_pair(filename):
    list_of_pair = []
    with open(filename, 'r') as f:
        for line in f:
            list_line = line.split('?')
            product1 = json.loads(list_line[2], encoding = 'latin-1')
            product2 = json.loads(list_line[4], encoding = 'latin-1')
            label = list_line[5].strip()
            product_pair = (product1, product2, label)
            list_of_pair.append(product_pair)
    return list_of_pair

if __name__ == '__main__':
    list_of_pair_20000 = generate_list_of_pair('elec_pairs_stage3.txt')
    with open('list_of_pair_20000.pickle', 'wb') as handle:
        pickle.dump(list_of_pair_20000, handle)