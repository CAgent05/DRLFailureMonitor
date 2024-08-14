import os
import torch
from os.path import dirname
from sklearn.metrics import accuracy_score
from util import TSC_multivariate_data_loader, OS_CNN_easy_use
import argparse


p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--dataset", type=str, default="BipedalWalkerHCAC")
p.add_argument('-n', '--nsteps', type=int, default=20)
p.add_argument('--alg', type=str, default="Todynet", help='the algorithm used for training')
args = p.parse_args()

dataset = args.dataset
n = args.nsteps

log_dir = './OS-CNN/log/'
dataset_path = dirname("./data/")
model_dir = './model/OS-CNN/'
print(dataset_path)

print('running at:', dataset)   
# load data
X_train, y_train, X_test, y_test = TSC_multivariate_data_loader(dataset_path, dataset, nsteps=n)
print(X_train.shape)
model = OS_CNN_easy_use(
    log_dir = log_dir, # the Result_log_folder
    model_dir = model_dir, # the model folder
    nsteps = n,
    dataset_name = dataset,           # dataset_name for creat log under Result_log_folder
    device = "cuda:0",                     # Gpu 
    max_epoch = args.epochs,                       # In our expirement the number is 2000 for keep it same with FCN for the example dataset 500 will be enough
    Max_kernel_size = 89, 
    start_kernel_size = 1,
    paramenter_number_of_layer_list = [8*128, (5*128*256 + 2*256*128)/2], 
    quarter_or_half = 4,
    )

model.fit(X_train, y_train, X_test, y_test)

y_predict = model.predict(X_test)

acc = accuracy_score(y_predict, y_test)
print(acc)
    