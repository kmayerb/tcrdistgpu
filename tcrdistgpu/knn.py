

import numpy as np
import pandas as pd
import os
import importlib.resources
from tcrdistgpu.distance import TCRgpu
from tqdm import tqdm
from scipy.sparse import dok_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def knn_tcr(tcr_train,
            tcr_test,
            label_train,
            label_test = None,
            chain = "a", mode = "cpu", kbest = 20,
            krange = range(1,21,2),
            adjust_class_weights = True,
            cdr3b_col = 'cdr3b',
            cdr3a_col = 'cdr3a',
            vb_col = 'vb',
            va_col = 'va',
            chunk_size =100, 
            tg = None):

  acc_store = list()
  auc_store = list()
  probs_store  = list()
  k_store = list()

  nrow = tcr_test.shape[0]
  ncol = tcr_train.shape[0]
  
  if tg is None:
    tg = TCRgpu(tcrs = tcr_train,
                tcrs2 = tcr_test,
                mode = mode,
                kbest = kbest,
                cdr3b_col = cdr3b_col,
                cdr3a_col = cdr3a_col ,
                vb_col = vb_col,
                va_col = va_col, 
                chunk_size = chunk_size)
  else:
    # Here we can input a precconfigured TCRgpu instance.
    tg.tcrs = tcr_train
    tg.tcrs2 = tcr_test
    tg.mode = mode
    tg.kbest = kbest
    #tg.cdr3b_col = cdr3b_col,
    #tg.cdr3a_col = cdr3a_col ,
    #tg.vb_col = vb_col,
    #tg.va_col = va_col, 
    tg.chunk_size = chunk_size

  print(f"--- Encoding TCRs as vectors")
  if chain == "a":
    encoded1 = tg.encode_tcrs_a(tcrs = tcr_test)
    encoded2 = tg.encode_tcrs_a(tcrs = tcr_train)
  if chain == "b":
    encoded1 = tg.encode_tcrs_b(tcrs = tcr_test)
    encoded2 = tg.encode_tcrs_b(tcrs = tcr_train)
  if chain == "ab":
    encoded1 = tg.encode_tcrs(tcrs = tcr_test)
    encoded2 = tg.encode_tcrs(tcrs = tcr_train)

  print(f"--- Computing TCRdistances between query x reference tcrs [({nrow})x({ncol})]")
  indices, distances = tg.compute(encoded1= encoded1, encoded2=encoded2)
  print(f"--- Retained column indices of {kbest} nearest neigbors for each row")
  print(f"--- Shape : {indices.shape}")

  if mode == "cuda":
    indices, distances = indices.get(), distances.get()
  
  #print(f"--- Converting to sparse matrix")
  # dok = dok_matrix((nrow, ncol), dtype=np.int16)
  # for i, jdx, in tqdm(enumerate(indices), total = nrow):
  #   jdist = distances[i]
  #   for j,d in zip(jdx,jdist):
  #     dok[i,j] = max(1,d)
  # print(f"--- Completed sparse matrix {dok.shape}))")

  labels = label_train
  y_test = label_test

  print(f"--- Performing kNN Classification")
  total_k = len([x for x in krange])
  for k in tqdm(krange, total = total_k):
    #print(k)
    k_store.append(k)

    # Compute weights using distances and labels
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(label_train), y=label_train)
    class_weight_dict = {cls: weight for cls, weight in zip(np.unique(labels), class_weights)}

    weights = np.zeros_like(distances, dtype = 'float32')
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            label = labels[indices[i, j]]
            if adjust_class_weights:
              weights[i, j] = class_weight_dict.get(label) / (distances[i, j] + 1e-5)
            else:
              weights[i, j] = 1 / (distances[i, j] + 1e-5)

    # Predict based on weighted votes
    weighted_votes = np.zeros((distances.shape[0], len(np.unique(labels))))
    for i in range(distances.shape[0]):
        for j in range(k):
            #import pdb; pdb.set_trace()
            label = labels[indices[i, j]]
            weighted_votes[i, label] += weights[i, j]

    # Select the class with the highest weighted vote
    predictions = np.argmax(weighted_votes, axis=1)
    p0 = (weighted_votes[:,0])/ (weighted_votes.sum(axis =1 ))
    p1 = (weighted_votes[:,1])/ (weighted_votes.sum(axis =1 ))
    probs_test = np.column_stack([p0,p1])
    probs_store.append(probs_test)
    # If test data is labeled, we can compute AUC
    if label_test is not None:
      acc_store.append( accuracy_score(label_test, predictions) )
      auc_store.append( roc_auc_score(label_test, probs_test[:,1]))

  if label_test is not None:
      print(f"---Returning AUC metric based on provided labels")
      df = pd.DataFrame({'Accuracy' : acc_store,
                  'AUC': auc_store,
                  'k':k_store,
                  'adjust_class_weights':adjust_class_weights,
                  'mode':mode})
  else:
    print(f"---Returning predictions only")
    df = pd.DataFrame({'k':k_store,
                         'adjust_class_weights':adjust_class_weights,
                         'mode':mode})

  return(df,probs_store )
