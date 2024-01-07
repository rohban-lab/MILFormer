### Some parts from https://github.com/binli123/dsmil-wsi ###
import os, argparse, pickle, itertools
from copy import deepcopy
from pathlib import Path

from sklearn.utils import shuffle
import numpy as np
import pandas as pd


def get_data(file_path):
    df = pd.read_csv(file_path)
    df = pd.DataFrame(df)
    df = df[df.columns[0]]
    data_list = []
    for i in range(0, df.shape[0]):
        data = str(df.iloc[i]).split(' ')
        ids = data[0].split(':')
        instance_id = int(ids[0])
        bag_id = int(ids[1])
        class_id = int(ids[2])
        data = data[1:]
        feature_vector = np.zeros(len(data))
        for i, feature in enumerate(data):
            feature_data = feature.split(':')
            if len(feature_data) == 2:
                feature_vector[i] = feature_data[1]
        data_list.append([instance_id, bag_id, class_id, feature_vector])
    return data_list


def get_bag(data, idb):
    data_array = np.array(data, dtype=object)
    bag_id = data_array[:, 1]
    return data_array[np.where(bag_id == idb)]


def find_admissible_shuffle(args, bag_ins_list):
    """
    Shuffle the bags until there's a shuffle, where each fold has both positive and negative bags in all splits
    bag_ins_list: [
            [bag_0_label (int), np.array([np.array(instance_0_embeddings), np.array(instance_0_embeddings), ...])]
            [bag_1_label (int), np.array([np.array(instance_0_embeddings), np.array(instance_0_embeddings), ...])]
            ...
        ]
    """
    found_valid_shuffle = False
    while not found_valid_shuffle:
        bag_ins_list = shuffle(bag_ins_list)
        for k in range(0, args.num_folds):
            train_ins_list, valid_ins_list, test_ins_list = cross_validation_set(
                bag_ins_list, num_folds=args.num_folds, current_fold=k, valid_ratio=args.train_valid_ratio
            )

            train_bags_labels = [np.clip(bag[0], 0, 1) for bag in train_ins_list]
            valid_bags_labels = [np.clip(bag[0], 0, 1) for bag in valid_ins_list]
            test_bags_labels = [np.clip(bag[0], 0, 1) for bag in test_ins_list]

            if not (0 in train_bags_labels and 1 in train_bags_labels):
                break
            if not (0 in valid_bags_labels and 1 in valid_bags_labels):
                break
            if not (0 in test_bags_labels and 1 in test_bags_labels):
                break
            found_valid_shuffle = True

    return bag_ins_list


def cross_validation_set(bag_ins_list, num_folds: int, current_fold: int, valid_ratio: float):
    csv_list = deepcopy(bag_ins_list)
    n = int(len(csv_list) / num_folds)

    chunked = [csv_list[i:i + n] for i in range(0, len(csv_list), n)]

    test_list = chunked.pop(current_fold)
    train_valid_list = list(itertools.chain.from_iterable(chunked))  # this should be after the popping!

    train_list = train_valid_list[0:int(len(train_valid_list) * (1 - valid_ratio))]
    valid_list = train_valid_list[int(len(train_valid_list) * (1 - valid_ratio)):]
    return train_list, valid_list, test_list


def main(args, datasets_base_path='./'):
    mil_dataset_registry = {
        'musk1': ('Musk', 'musk1norm.svm', 166),
        'musk2': ('Musk', 'musk2norm.svm', 166),
        'elephant': ('Elephant', 'data_100x100.svm', 230),
        'fox': ('Fox', 'data_100x100.svm', 230),
        'tiger': ('Tiger', 'data_100x100.svm', 230),
    }
    dataset_folder, dataset_file, args.feats_size = mil_dataset_registry[args.dataset]
    data_all = get_data(os.path.join(datasets_base_path, dataset_folder, dataset_file))

    bag_ins_list = []
    num_bag = data_all[-1][1] + 1
    for i in range(num_bag):
        bag_data = get_bag(data_all, i)
        bag_label = bag_data[0, 2]
        bag_vector = bag_data[:, 3]
        bag_ins_list.append([bag_label, bag_vector])

    bag_ins_list = find_admissible_shuffle(args, bag_ins_list)
    file_name = f'{Path(dataset_file).stem}_{args.num_folds}folds_{args.train_valid_ratio}split.pkl'
    with open(os.path.join(datasets_base_path, dataset_folder, file_name), 'wb') as f:
        pickle.dump(bag_ins_list, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIL Dataset Cross-Validation')
    parser.add_argument('--dataset', default='musk1', type=str,
                        help='Choose MIL datasets from: musk1, musk2, elephant, fox, tiger [musk1]')
    parser.add_argument('--num_folds', default=10, type=int, help='Number of cross validation fold [10]')
    parser.add_argument('--train_valid_ratio', default=0.2, type=float, help='Train/Valid ratio')
    args = parser.parse_args()

    main(args)
