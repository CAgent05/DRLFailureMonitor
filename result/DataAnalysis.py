import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, auc
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Data processing')
parser.add_argument('-d', '--dataset', metavar='DATASET', default='BipedalWalkerHC')
parser.add_argument('-n', '--nsteps', type=int, default=20)
parser.add_argument('--alg', type=str, default="Todynet", help='the algorithm used for training')
args = parser.parse_args()


path = 'result/' + args.alg + '/' + args.dataset + '_' + str(args.nsteps) + '.csv'
data = pd.read_csv(path)
print(path)

df = pd.DataFrame(columns=['tp', 'fp', 'fn', 'tn', 'accuracy', 'f1', 'precision', 'recall', 'FPR', 'Steps', 'MCC', 'G-Measure', 'balance',' G-mean'])


pre = data['Pre'].to_list()
label = data['True'].to_numpy()
prob = data['Probabilities'].to_numpy()
index = np.where(label == 1)
steps = data['Steps'].to_numpy()[index]
print(steps)
print(np.where(label != pre))
# steps = sum(steps) / len(steps)
index = np.where(steps < 200)
steps = sum(steps[index])/ len(steps[index])
print('steps', steps)

tn, fp, fn, tp = confusion_matrix(label, pre).ravel()
accuracy = accuracy_score(label, pre)
f1 = f1_score(label, pre)
precision = precision_score(label, pre)
recall = recall_score(label, pre)
FPR = fp / (fp + tn)
MCC = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
G_mean = np.sqrt(recall * (1 - FPR))
balance = 1 - np.sqrt((1-FPR) ** 2 + (1-recall) ** 2) / np.sqrt(2)
G_measure = 2 * recall * (1 - FPR) / (recall + (1 - FPR))
f1 = f1_score(label, pre)
print('-------------' + '-------------')
print('tn', tn, 'fp', fp, 'fn', fn, 'tp', tp)
print('f1', f1, 'FPR', FPR, 'precision', precision, 'recall', recall, 'accuracy', accuracy, 'steps', steps, 'MCC', MCC, 'G-Measure', G_measure, 'balance', balance, 'G-mean', G_mean)
df.loc[len(df)] = [tp, fp, fn, tn, accuracy, f1, precision, recall, FPR, steps, MCC, G_measure, balance, G_mean]

precision, recall, thresholds = precision_recall_curve(label, prob)
print(precision,recall,thresholds)
auc = auc(recall, precision)
print('AUC', auc)

f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold_index = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_index]
best_f1_score = f1_scores[best_threshold_index]
print('Best Threshold', best_threshold, 'Best F1 Score', best_f1_score)

print(df)
