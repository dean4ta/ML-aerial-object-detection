# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 13:07:07 2018

@author: dean4ta
"""
from sklearn.metrics import confusion_matrix

'''
    INDICES OF LABELS
    0 : White Car
    1 : Red Car
    2 : Pool
    3 : Pond
'''

def analyzeConfusionMatrix(l_true, l_pred, index_of_label):
    C_matrix = confusion_matrix(l_true, l_pred)
    TP = C_matrix[index_of_label, index_of_label]
    FP = sum(C_matrix[:,index_of_label]) - TP
    FN = sum(C_matrix[index_of_label,:]) - TP
    TN = sum(sum(C_matrix)) - TP - FP - FN
    
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    Bethesda = FP/(FP+TN) #Fallout
    
    return TP,FP,FN,TN,Precision,Recall,Bethesda
