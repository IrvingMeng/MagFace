#!/usr/bin/env python

import argparse
import os
import cv2
import math
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate

# basic args
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--feat-list', type=str,
                    help='The cache folder for validation report')
parser.add_argument('--pair-list', type=str,
                    help='whether the img in feature list is same person')
parser.add_argument('--eval-type', type=str,
                    default='1v1',
                    help='The evaluation type')
parser.add_argument('--distance-metric', type=int,
                    default=1,
                    help='0: Euclidian Distance. \
                          1: Cosine Distance.')
parser.add_argument('--test-folds', type=int,
                    default=10,
                    help='')                    

def load_feat_pair(feat_path, pair_path):
    pairs = {}
    with open(pair_path, 'r') as f:
        is_pair = list(map(lambda item: item.strip().split()[-1], f.readlines()))
    with open(feat_path) as f:
        ls = f.readlines()
        for idx in range(len(is_pair)):
            feat_a = ls[idx*2]
            feat_b = ls[idx*2+1]
            is_same = is_pair[idx]
            pairs[idx] = [feat_a, feat_b, is_same]
    return pairs


def perform_1v1_eval(args):
    # load features
    feat_list = args.feat_list
    pair_list = args.pair_list
    feat_pairs = load_feat_pair(feat_list, pair_list)
    # ensemble feats
    embeddings0 = []
    embeddings1 = []
    targets = []
    

    for k, v in feat_pairs.items():
        feat_a = v[0]
        feat_b = v[1]
        ab_is_same = int(v[2])

        # convert into np
        np_feat_a = np.asarray(feat_a.split()[1:513], dtype=float)
        np_feat_b = np.asarray(feat_b.split()[1:513], dtype=float)
        # append
        embeddings0.append(np_feat_a)
        embeddings1.append(np_feat_b)

        targets.append(ab_is_same)

    # evaluate
    embeddings0 = np.vstack(embeddings0)
    embeddings1 = np.vstack(embeddings1)
    targets = np.vstack(targets).reshape(-1,)

    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, accuracy = calculate_roc(
            thresholds, embeddings0, embeddings1,targets,
            nrof_folds=args.test_folds, subtract_mean=True)
    print('    Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))

def distance_(embeddings0, embeddings1):
    # Distance based on cosine similarity
    dot = np.sum(np.multiply(embeddings0, embeddings1), axis=1)
    norm = np.linalg.norm(embeddings0, axis=1) * np.linalg.norm(embeddings1, axis=1)
    # shaving
    similarity = np.clip(dot / norm, -1., 1.)
    dist = np.arccos(similarity) / math.pi
    return dist

def calculate_roc(thresholds, embeddings0, embeddings1, 
                  actual_issame, nrof_folds=10, subtract_mean=False):
    assert(embeddings0.shape[0] == embeddings1.shape[0])
    assert(embeddings0.shape[1] == embeddings1.shape[1])

    nrof_pairs = min(len(actual_issame), embeddings0.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings0[train_set], embeddings1[train_set]]), axis=0)
        else:
            mean = 0.

        dist = distance_(embeddings0-mean, embeddings1-mean)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

class LFold:
    def __init__(self, n_splits = 2, shuffle = False):
        self.n_splits = n_splits
        if self.n_splits>1:
            self.k_fold = KFold(n_splits = n_splits, shuffle = shuffle)

    def split(self, indices):
        if self.n_splits>1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]

def main():
    args = parser.parse_args()
    perform_1v1_eval(args)

if __name__ == '__main__':
    main()


