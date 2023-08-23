import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np

def perf_measure(y_pred, y_actual):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    y_actual = y_actual.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().argmax(dim=1).numpy()


    for i in range(len(y_actual)): 
        if y_pred[i]==y_actual[i]==1:
           TP += 1
        if y_pred[i]==1 and y_pred[i]!=y_actual[i]:
           FP += 1
        if y_pred[i]==y_actual[i]==0:
           TN += 1
        if y_pred[i]==0 and y_pred[i]!=y_actual[i]:
           FN += 1
    
    
    specificity = TN/(TN+FP+pow(10,-15))
    return specificity
    
def perf_measure_sen(y_pred, y_actual):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    y_actual = y_actual.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().argmax(dim=1).numpy()


    for i in range(len(y_actual)): 
        if y_pred[i]==y_actual[i]==1:
           TP += 1
        if y_pred[i]==1 and y_pred[i]!=y_actual[i]:
           FP += 1
        if y_pred[i]==y_actual[i]==0:
           TN += 1
        if y_pred[i]==0 and y_pred[i]!=y_actual[i]:
           FN += 1
    
    
    sensitivity = TP/(TP+FN+pow(10,-15))
    return sensitivity
    
def auc_score(y_pred,y_actual):
    
    y_actual = y_actual.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().argmax(dim=1).numpy()
    
    fpr,tpr,thresholds = metrics.roc_curve(y_actual,y_pred)
    score = metrics.auc(fpr,tpr)
    return score

def MAE(scores, targets):
    MAE = F.l1_loss(scores, targets)
    MAE = MAE.detach().item()
    return MAE


def accuracy_TU(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    return acc


def accuracy_MNIST_CIFAR(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    return acc

def accuracy_CITATION_GRAPH(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    acc = acc / len(targets)
    return acc


def accuracy_SBM(scores, targets):
    S = targets.cpu().numpy()
    C = np.argmax( torch.nn.Softmax(dim=1)(scores).cpu().detach().numpy() , axis=1 )
    CM = confusion_matrix(S,C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets==r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r,r]/ float(cluster.shape[0])
            if CM[r,r]>0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = 100.* np.sum(pr_classes)/ float(nb_classes)
    return acc


def binary_f1_score(scores, targets):
    """Computes the F1 score using scikit-learn for binary class labels. 
    
    Returns the F1 score for the positive class, i.e. labelled '1'.
    """
    y_true = targets.cpu().numpy()
    y_pred = scores.argmax(dim=1).cpu().numpy()
    return f1_score(y_true, y_pred, average='binary')

  
def accuracy_VOC(scores, targets):
    scores = scores.detach().argmax(dim=1).cpu()
    targets = targets.cpu().detach().numpy()
    acc = f1_score(scores, targets, average='weighted')
    return acc


def accuracy_WikiCS(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    acc = acc / len(targets)
    return acc