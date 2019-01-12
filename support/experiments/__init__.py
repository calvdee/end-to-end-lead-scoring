from sklearn.metrics import roc_auc_score, make_scorer

def get_auc_scorer():
  scorer = make_scorer(roc_auc_score)
  return scorer