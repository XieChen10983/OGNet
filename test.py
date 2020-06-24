from __future__ import print_function
import numpy as np
from sklearn import metrics
import re
lastNum = re.compile(r'(?:[^\d]*(\d+)[^\d]*)+')
import torch
from dataloader import load_data
from model import OGNet
from opts import parse_opts


def check_auc(g_model_path, d_model_path, opt, i):
    opt.batch_shuffle = False
    opt.drop_last = False
    dataloader = load_data(opt)
    model = OGNet(opt, dataloader)
    if torch.cuda.is_available():
        model.cuda()
    d_results, labels = model.test_patches(g_model_path, d_model_path, i)
    d_results = np.concatenate(d_results)
    labels = np.concatenate(labels)
    fpr1, tpr1, thresholds1 = metrics.roc_curve(labels, d_results, pos_label=1)  # (y, score, positive_label)
    fnr1 = 1 - tpr1
    eer_threshold1 = thresholds1[np.nanargmin(np.absolute((fnr1 - fpr1)))]
    eer_threshold1 = eer_threshold1
    d_f1 = np.copy(d_results)
    d_f1[d_f1 >= eer_threshold1] = 1
    d_f1[d_f1 < eer_threshold1] = 0
    f1_score = metrics.f1_score(labels, d_f1, pos_label=0)
    print("AUC: {0}, F1_score: {1}".format(metrics.auc(fpr1, tpr1), f1_score))


if __name__ == '__main__':
    Opt = parse_opts()
    Opt.data_path = './data/test/'  # test data path
    G_model_path = './models/phase_two_g'  # generator model path
    D_model_path = './models/phase_two_d'  # discriminator model path
    print('working on :', G_model_path, D_model_path)
    check_auc(G_model_path, D_model_path, Opt, 0)
