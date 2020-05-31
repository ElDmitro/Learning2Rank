import itertools
import re
import sys
import numpy as np
import pandas as pd


def pairwise_transform(documents, relevants):
    """
    Transform documents data into pair-wise form
    
    
    Parameters
    ----------
    documents : array, shape (n_samples, n_features)
        The data
    relevants : array, shape (n_samples,)
        Target labels.
    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    """
    
    X_trans = []
    y_trans = []
    
    pairs_grid = itertools.combinations(range(documents.shape[0]), 2)
    for pid, (i, j) in enumerate(pairs_grid):
        y_left, y_right = relevants[i], relevants[j]
        if y_left == y_right:
            continue
            
        X_trans.append(
            documents[j] - documents[i]
        )
        y_trans.append(
            np.sign(y_left - y_right)
        )
        
    return np.array(X_trans), np.array(y_trans)


def construct_pairwise(queries, features):
    """
    Transform queries table into pair-wise form
    
    
    Parameters
    ----------
    queries : DataFrame
        Query - Document table
    features : array, shape (n_features,)
        Features labels
    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    """
    
    queries_grid = queries.qid.unique()
    
    X_trans = []
    y_trans = []
    for qid in queries_grid:
        mask = queries.qid == qid
        
        X = queries.loc[mask, features].values
        y = queries.loc[mask, 'relevent_val'].values
        
        X, y = pairwise_transform(X, y)
        
        if X.shape[0] == 0:
            continue
        if X.ndim == 1:
            X = X[np.newaxis]
            
        X_trans.append(X)
        y_trans.append(y)
        
    return np.vstack(X_trans), np.hstack(y_trans)


def group_counts(arr):
    d = np.ones(arr.size, dtype=int)
    d[1:] = (arr[:-1] != arr[1:]).astype(int)
    return np.diff(np.where(np.append(d, 1))[0])


def group_offsets(arr):
    """Return a sequence of start/end offsets for the value subgroups in the input"""
    d = np.ones(arr.size, dtype=int)
    d[1:] = (arr[:-1] != arr[1:]).astype(int)
    idx = np.where(np.append(d, 1))[0]
    return zip(idx, idx[1:])


def load_docno(fname, letor=False):
    """Load docnos from the input in the SVMLight format"""
    if letor:
        docno_pattern = re.compile(r'#\s*docid\s*=\s*(\S+)')
    else:
        docno_pattern = re.compile(r'#\s*(\S+)')

    docno = []
    for line in open(fname):
        if line.startswith('#'):
            continue
        m = re.search(docno_pattern, line)
        if m is not None:
            docno.append(m.group(1))
    return np.array(docno)


def print_trec_run(qid, docno, pred, run_id='exp', output=None):
    """Print TREC-format run to output"""
    if output is None:
        output = sys.stdout
    for a, b in group_offsets(qid):
        idx = np.argsort(-pred[a:b]) + a  # note the minus and plus a
        for rank, i in enumerate(idx, 1):
            output.write('{qid} Q0 {docno} {rank} {sim} {run_id}\n'.
                         format(qid=qid[i], docno=docno[i], rank=rank, sim=pred[i], run_id=run_id))
