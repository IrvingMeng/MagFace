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
from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


# the ijbc dataset is from insightface
# using the cos similarity
# no flip test

# basic args
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--feat_list', type=str,
                    help='The cache folder for validation report')
parser.add_argument('--base_dir', default='data/IJBC/')
parser.add_argument('--type', default='c')
parser.add_argument('--embedding_size', default=512, type=int)
parser.add_argument('--magface_qlt', default=0, type=int)


def read_template_media_list(path):
    ijb_meta, templates, medias = [], [], []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split(' ')
        ijb_meta.append(parts[0])
        templates.append(int(parts[1]))
        medias.append(int(parts[2]))
    return np.array(templates), np.array(medias)


def read_template_pair_list(path):
    t1, t2, label = [], [], []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        data = line.strip().split(' ')
        t1.append(int(data[0]))
        t2.append(int(data[1]))
        label.append(int(data[2]))
    return np.array(t1), np.array(t2), np.array(label)


def read_feats(args):
    with open(args.feat_list, 'r') as f:
        lines = f.readlines()
    img_feats = []
    for line in lines:
        data = line.strip().split(' ')
        img_feats.append([float(ele) for ele in data[1:1+args.embedding_size]])
    img_feats = np.array(img_feats).astype(np.float32)
    return img_feats


def image2template_feature(img_feats=None,
                           templates=None,
                           medias=None,
                           magface_qlt=None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates = np.unique(templates)
    # template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))
    template_feats = torch.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]

        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(
            face_medias, return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            media_norm_feats += [np.mean(face_norm_feats[ind_m],
                                         0, keepdims=False)]

        # media_norm_feats = np.array(media_norm_feats)
        media_norm_feats = torch.tensor(media_norm_feats)
        if magface_qlt != 1:
            # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
            media_norm_feats = F.normalize(media_norm_feats)
        # template_feats[count_template] = np.mean(media_norm_feats, 0)
        template_feats[count_template] = torch.mean(media_norm_feats, 0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(count_template))
    # template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    return template_feats, unique_templates


def distance_(embeddings0, embeddings1):
    # # Distance based on cosine similarity
    # dot = np.sum(np.multiply(embeddings0, embeddings1), axis=1)
    # norm = np.linalg.norm(embeddings0, axis=1) * np.linalg.norm(embeddings1, axis=1)
    # # shaving
    # similarity = np.clip(dot / norm, -1., 1.)
    # dist = np.arccos(similarity) / math.pi
    # return dist
    cos = nn.CosineSimilarity(dim=1, eps=0)
    simi = torch.clamp(cos(embeddings0, embeddings1), min=-1, max=1)
    dist = torch.acos(simi)/math.pi
    return dist.cpu().numpy()


def verification(template_feats, unique_templates, p1=None, p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates)+1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    scores = np.zeros((len(p1),))   # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    # small batchsize instead of all pairs in one batch due to the memory limiation
    batchsize = 100000
    sublists = [total_pairs[i:i + batchsize]
                for i in range(0, len(p1), batchsize)]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_feats[template2id[p1[s]]]
        feat2 = template_feats[template2id[p2[s]]]
        similarity_score = distance_(feat1, feat2)
        scores[s] = 1-similarity_score
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return scores


def perform_verification(args):
    # load the data
    templates, medias = read_template_media_list(
        '{}/meta/ijb{}_face_tid_mid.txt'.format(args.base_dir, args.type)
    )
    p1, p2, label = read_template_pair_list(
        '{}/meta/ijb{}_template_pair_label.txt'.format(
            args.base_dir, args.type)
    )
    img_feats = read_feats(args)

    # calculate scores
    template_feats, unique_templates = image2template_feature(img_feats,
                                                              templates,
                                                              medias,
                                                              args.magface_qlt)
    scores = verification(template_feats, unique_templates, p1, p2)

    # show the results
    print('IJB{} 1v1 verification:\n'.format(args.type))
    fpr, tpr, _ = roc_curve(label, scores)
    roc_auc = auc(fpr, tpr)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)  # select largest tpr at same fpr
    tpr_fpr_row = []
    x_labels = [10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
    to_print = ''
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(
            list(zip(abs(fpr-x_labels[fpr_iter]), range(len(fpr)))))
        # print('     {} TAR.FAR{}'.format(tpr[min_index], x_labels[fpr_iter]))
        print('  {:0.4f}'.format(tpr[min_index]))
        to_print = to_print + '  {:0.4f}'.format(tpr[min_index])
    print(to_print)


def perform_recognition(args):
    pass


def main():
    args = parser.parse_args()
    perform_verification(args)
    perform_recognition(args)


if __name__ == '__main__':
    main()


"""
score_save_path = './IJBC/result'
files = glob.glob(score_save_path + '/MS1MV2*.npy')  
methods = []
scores = []
for file in files:
    methods.append(Path(file).stem)
    scores.append(np.load(file)) 
methods = np.array(methods)
scores = dict(zip(methods,scores))
colours = dict(zip(methods, sample_colours_from_colourmap(methods.shape[0], 'Set2')))
#x_labels = [1/(10**x) for x in np.linspace(6, 0, 6)]
x_labels = [10**-6, 10**-5, 10**-4,10**-3, 10**-2, 10**-1]
tpr_fpr_table = PrettyTable(['Methods'] + map(str, x_labels))
fig = plt.figure()
for method in methods:
    fpr, tpr, _ = roc_curve(label, scores[method])
    roc_auc = auc(fpr, tpr)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr) # select largest tpr at same fpr
    plt.plot(fpr, tpr, color=colours[method], lw=1, label=('[%s (AUC = %0.4f %%)]' % (method.split('-')[-1], roc_auc*100)))
    tpr_fpr_row = []
    tpr_fpr_row.append(method)
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(list(zip(abs(fpr-x_labels[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append('%.4f' % tpr[min_index])
    tpr_fpr_table.add_row(tpr_fpr_row)
plt.xlim([10**-6, 0.1])
plt.ylim([0.3, 1.0])
plt.grid(linestyle='--', linewidth=1)
plt.xticks(x_labels) 
plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True)) 
plt.xscale('log')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC on IJB-C')
plt.legend(loc="lower right")
plt.show()
#fig.savefig('IJB-B.pdf')
"""
