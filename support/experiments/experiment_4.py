import pandas as pd
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion

NUMERIC_FEATURES = [
    'age', 
    'campaign', 
    'previous', 
    'emp.var.rate', 
    'cons.price.idx', 
    'cons.conf.idx', 
    'euribor3m', 
    'nr.employed'
]

CATEGORICAL_FEATURES =  [
  'job',
  'marital',
  'education',
  'default',
  'housing',
  'loan',
  'contact',
  'month',
  'day_of_week',
  'poutcome'
]

def ft_job(X):
  job = X.copy()
  job[job.isin(['retired', 'student', 'unemployed'])] = 'rare_high_prob'
  return job.values.reshape(-1,1)

def ft_month(X):
  month = X.copy()
  month[month.isin(['mar', 'oct', 'dec', 'sep'])] = 'rare_high_prob'
  return month.values.reshape(-1,1)

def ft_poutcome(X):
  poutcome = X.copy()
  poutcome[poutcome.isin(['failure', 'nonexistent'])] = 'failure'
  return poutcome.values.reshape(-1,1)

def ft_pdays(X):
  pdays = X.apply(lambda x: 1 if x != 999 else 0)
  return pdays.values.reshape(-1,1)

def get_categorical_ct():
  add_job = FunctionTransformer(ft_job, validate=False)
  add_month = FunctionTransformer(ft_month, validate=False)
  add_poutcome = FunctionTransformer(ft_poutcome, validate=False)
  add_pdays = FunctionTransformer(ft_pdays, validate=False)

  features = set(CATEGORICAL_FEATURES) - set(['job', 'month', 'poutcome'])
  features = list(features)

  cat_features = [
    ('categoricals', 'passthrough', features),
    ('add_job', add_job, 'job'),
    ('add_month', add_month, 'month'),
    ('add_poutcome', add_poutcome, 'poutcome'),
    ('add_pdays', add_pdays, 'pdays')
  ]
  cat_ct = ColumnTransformer(cat_features)
  
  return cat_ct

def get_categorical_pipeline():
  cat_cts = get_categorical_ct()

  # Create the pipeline to transform categorical features
  cat_pipeline = Pipeline([
    ('cat_ct', cat_cts),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
  ])

  return cat_pipeline


def get_pipeline():
  pipeline = get_categorical_pipeline()
  return pipeline

def baseline_model_predictions(X, y, n_targeted):
  # Get all of the instances where the previous campaign was a success
  success = X[X.poutcome == 'success']
  
  # Calcuate how many more instances we need
  n_rest = n_targeted - len(success)
  
  # Randomly choose from the remaining instances
  rest = X[~(X.index.isin(success.index))].sample(n=n_rest, random_state=1)
  
  # Combine the targeted and random groups
  baseline_targets = pd.concat([success, rest], axis=0)
  baseline_ys = y.loc[baseline_targets.index]

  return baseline_ys