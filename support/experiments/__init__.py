import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.metrics import classification_report
from support.parameters import AVG_COST, AVG_REVENUE, P_TARGETED

def get_auc_scorer():
  scorer = make_scorer(roc_auc_score)
  return scorer

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

def display_model_performance(model, X_test, y_test):
  n_targeted = int(len(X_test) * P_TARGETED)
  print('Number of targeted customers:', n_targeted)

  baseline_preds = baseline_model_predictions(X_test, y_test, n_targeted)
  baseline_outcomes = baseline_preds.apply(lambda x: AVG_COST if x == 0 else AVG_COST + AVG_REVENUE)
  baseline_profit = sum(baseline_outcomes)
  baseline_conv_rate = baseline_preds.sum() / len(baseline_preds)
  base_rate = y_test.sum() / len(y_test)

  X_test_trans = model.pipeline.transform(X_test)

  preds = model.model.predict(X_test_trans)
  probs = model.model.predict_proba(X_test_trans)

  probs_preds_mat = np.hstack([probs, y_test.values.reshape(-1,1), preds.reshape(-1,1)])

  # Create a dataframe of probabilities and actual / predicted outcomes
  probs_df = pd.DataFrame(probs_preds_mat, columns=['p_no', 'p_yes', 'actual', 'predicted'])

  # Sort customers by the probability that they will convert
  model_targets = probs_df.sort_values('p_yes', ascending=False)

  # Take the top N
  model_targets = model_targets.head(n_targeted)

  # Calculate financial outcomes
  model_outcomes = model_targets.actual.apply(lambda x: AVG_COST if x == 0 else AVG_COST + AVG_REVENUE)
  model_profit = sum(model_outcomes)
  model_conv_rate = model_targets.actual.sum() / len(model_targets)

  model_profit_lift_baseline = model_profit / baseline_profit, model_profit - baseline_profit
  model_conv_lift_baseline = model_conv_rate / baseline_conv_rate

  print('Model profit: ${:,}'.format(model_profit))
  print('Model lift over basline profit: {:.1f} or ${:,}'.format(model_profit_lift_baseline[0] ,model_profit_lift_baseline[1]))
  print('Targeted conversion rate: {:.2f}'.format(model_conv_rate))
  print('Conversion rate lift over baseline: {:.2f}'.format(model_conv_lift_baseline))
  print('')
  print(classification_report(model_targets.actual, model_targets.predicted))