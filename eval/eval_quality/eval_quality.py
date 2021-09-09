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
import matplotlib.pyplot as plt

# basic args
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--feat-list', type=str,
                    help='The feature list')
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


def calc_quality(line):
    "calcualte the magnitude from a feature line"
    feat = np.asarray(line.split()[1:513], dtype=np.float64)
    return np.linalg.norm(feat)
    

def load_feat_pair(feat_path, pair_path):
    pairs = {}
    with open(pair_path, 'r') as f:
        is_pair = list(map(lambda item: item.strip().split()[-1], f.readlines()))
    with open(feat_path) as f:
        ls = f.readlines()
        for idx in range(len(is_pair)):
            feat_a = ls[idx*2]
            feat_b = ls[idx*2+1]
            qlt = min(calc_quality(feat_a), calc_quality(feat_b))
            is_same = is_pair[idx]
            pairs[idx] = [feat_a, feat_b, is_same, qlt]
    return pairs


def distance_(embeddings0, embeddings1):
    # Distance based on cosine similarity
    dot = np.sum(np.multiply(embeddings0, embeddings1), axis=1)
    norm = np.linalg.norm(embeddings0, axis=1) * np.linalg.norm(embeddings1, axis=1)
    # shaving
    similarity = np.clip(dot / norm, -1., 1.)
    dist = np.arccos(similarity) / math.pi
    return dist


def calc_score(embeddings0, embeddings1, actual_issame, subtract_mean=False):
    assert(embeddings0.shape[0] == embeddings1.shape[0])
    assert(embeddings0.shape[1] == embeddings1.shape[1])

    if subtract_mean:
        mean = np.mean(np.concatenate([embeddings0, embeddings1]), axis=0)
    else:
        mean = 0.
    dist = distance_(embeddings0, embeddings1)
    # sort in a desending order
    pos_scores = np.sort(dist[actual_issame==1])
    neg_scores = np.sort(dist[actual_issame==0])
    return pos_scores, neg_scores


def save_pdf(fnmrs_list, unconsidered_rates, args):
    colors = ['red', 'blue', 'green', 'black']
    method_labels = ['A', 'B', 'C', 'D']
    plt.figure(figsize=(8,4))
    plt.plot(unconsidered_rates[:len(fnmrs_list)], fnmrs_list, color=colors[0])    
    plt.xlabel('Ratio of unconsidered image [%]')
    plt.ylabel('FNMR')
    plt.title('FNMR at 0.001 FMR')
    
    plt.savefig('test.png')

         
def perform_1v1_quality_eval(args):
    # load features
    feat_pairs = load_feat_pair(args.feat_list, args.pair_list)
    
    # ensemble feats
    embeddings0, embeddings1, targets, qlts = [], [], [], []
    pair_qlt_list = [] # store the min qlt 
    for k, v in feat_pairs.items():
        feat_a = v[0]
        feat_b = v[1]
        ab_is_same = int(v[2])
        qlt = v[3]
        # convert into np
        np_feat_a = np.asarray(feat_a.split()[1:513], dtype=np.float64)
        np_feat_b = np.asarray(feat_b.split()[1:513], dtype=np.float64)
        # append
        embeddings0.append(np_feat_a)
        embeddings1.append(np_feat_b)
        targets.append(ab_is_same)
        qlts.append(qlt)

    # evaluate
    embeddings0 = np.vstack(embeddings0)
    embeddings1 = np.vstack(embeddings1)
    targets = np.vstack(targets).reshape(-1,)
    qlts = np.array(qlts)
    qlts_sorted_idx = np.argsort(qlts)        
    
    num_pairs = len(targets)
    unconsidered_rates = np.arange(0, 0.98, 0.05)
    fnmrs_list = []
    for u_rate in unconsidered_rates:
        hq_pairs_idx = qlts_sorted_idx[int(u_rate*num_pairs):]
        pos_dists, neg_dists = calc_score(embeddings0[hq_pairs_idx],
                                          embeddings1[hq_pairs_idx],
                                          targets[hq_pairs_idx])
        
        fmr = 1
        idx = len(neg_dists) - 1
        num_query = len(pos_dists)
        while idx >= 0:
            thresh = neg_dists[idx]
            num_acc = sum(pos_dists < thresh)
            fnmr = 1.0 * (num_query - num_acc) / num_query

            if fmr == 1e-3:
                sym = ' ' if thresh >= 0 else ''
                line = 'FNMR = {:.10f}  :  FMR = {:.10f}  :  THRESHOLD = {}{:.10f}'.format(fnmr, fmr, sym, thresh)
                print(line)
                fnmrs_list.append(fnmr)

            if idx == 0:
                break
            idx /= 10
            idx = int(idx)
            fmr /= float(10) 
    print(fnmrs_list)
    # np.save(args.quality_list.replace('.list', '.npy'), np.array(fnmrs_list))
    save_pdf(fnmrs_list, 100 * unconsidered_rates, args)       


def main():
    args = parser.parse_args()
    perform_1v1_quality_eval(args)

if __name__ == '__main__':
    main()
