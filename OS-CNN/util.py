from sklearn import preprocessing
from PIL import Image
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import os
import math
import copy
from sklearn.metrics import accuracy_score
from os.path import dirname
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import gymnasium as gym
from stable_baselines3 import PPO, SAC

def replace_nan_with_row_mean(a):
    out = np.where(np.isnan(a), ma.array(a, mask=np.isnan(a)).mean(axis=1)[:, np.newaxis], a)
    return np.float32(out)

def replace_nan_with_near_value(a):
    mask = np.isnan(a)
    idx = np.where(~mask,np.arange(mask.shape[1]),0)
    np.maximum.accumulate(idx,axis=1, out=idx)
    out = a[np.arange(idx.shape[0])[:,None], idx]
    return np.float32(out)

def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a

def TSC_data_loader(dataset_path,dataset_name):
    Train_dataset = np.loadtxt(
        dataset_path + '/' + dataset_name + '/' + dataset_name + '_TRAIN.tsv')
    Test_dataset = np.loadtxt(
        dataset_path + '/' + dataset_name + '/' + dataset_name + '_TEST.tsv')
    Train_dataset = Train_dataset.astype(np.float32)
    Test_dataset = Test_dataset.astype(np.float32)

    X_train = Train_dataset[:, 1:]
    y_train = Train_dataset[:, 0:1]

    X_test = Test_dataset[:, 1:]
    y_test = Test_dataset[:, 0:1]
    le = preprocessing.LabelEncoder()
    le.fit(np.squeeze(y_train, axis=1))
    y_train = le.transform(np.squeeze(y_train, axis=1))
    y_test = le.transform(np.squeeze(y_test, axis=1))
    return replace_nan_with_row_mean(X_train), y_train, replace_nan_with_row_mean(X_test), y_test

def fill_out_with_Nan(data,max_length):
    #via this it can works on more dimensional array
    pad_length = max_length-data.shape[-1]
    if pad_length == 0:
        return data
    else:
        pad_shape = list(data.shape[:-1])
        pad_shape.append(pad_length)
        Nan_pad = np.empty(pad_shape)*np.nan
        return np.concatenate((data, Nan_pad), axis=-1)
    

def get_label_dict(file_path):
    label_dict ={}
    with open(file_path) as file:
        lines = file.readlines()
        for line in lines:
            if '@classLabel' in line:
                label_list = line.replace('\n','').split(' ')[2:]
                for i in range(len(label_list)):
                    label_dict[label_list[i]] = i 
                
                break
    return label_dict


def get_data_and_label_from_ts_file(file_path,label_dict):
    with open(file_path) as file:
        lines = file.readlines()
        Start_reading_data = False
        Label_list = []
        Data_list = []
        max_length = 0
        for line in lines:
            if Start_reading_data == False:
                if '@data'in line:
                    Start_reading_data = True
            else:
                temp = line.split(':')
                Label_list.append(label_dict[temp[-1].replace('\n','')])
                data_tuple= [np.expand_dims(np.fromstring(channel, sep=','), axis=0) for channel in temp[:-1]]
                max_channel_length = 0
                for channel_data in data_tuple:
                    if channel_data.shape[-1]>max_channel_length:
                        max_channel_length = channel_data.shape[-1]
                data_tuple = [fill_out_with_Nan(data,max_channel_length) for data in data_tuple]
                data = np.expand_dims(np.concatenate(data_tuple, axis=0), axis=0)
                Data_list.append(data)
                if max_channel_length>max_length:
                    max_length = max_channel_length
        
        Data_list = [fill_out_with_Nan(data,max_length) for data in Data_list]
        X =  np.concatenate(Data_list, axis=0)
        Y =  np.asarray(Label_list)
        
        return np.float32(X), Y




def TSC_multivariate_data_loader(dataset_path, dataset_name, nsteps):
    
    path = dataset_path + '/Train/' + dataset_name + '_' + str(nsteps)
    
    X_train = torch.load(path + '/X_train.pt').squeeze(1).detach().cpu().numpy()
    y_train = torch.load(path + '/y_train.pt').detach().cpu().numpy()
    X_valid = torch.load(path + '/X_valid.pt').squeeze(1).detach().cpu().numpy()
    y_valid = torch.load(path + '/y_valid.pt').detach().cpu().numpy()
    
    return X_train, y_train, X_valid, y_valid

def eval_condition(iepoch,print_result_every_x_epoch):
    if (iepoch + 1) % print_result_every_x_epoch == 0:
        return True
    else:
        return False


def eval_model(model, dataloader):
    predict_list = np.array([])
    label_list = np.array([])
    for sample in dataloader:
        y_predict = model(sample[0])
        y_predict = y_predict.detach().cpu().numpy()
        y_predict = np.argmax(y_predict, axis=1)
        predict_list = np.concatenate((predict_list, y_predict), axis=0)
        label_list = np.concatenate((label_list, sample[1].detach().cpu().numpy()), axis=0)
    acc = accuracy_score(predict_list, label_list)
    return acc


def save_to_log(sentence, Result_log_folder, dataset_name):

    path = Result_log_folder + '/' + dataset_name + '.txt'
    print(path)
    with open(path, "a") as myfile:
        myfile.write(sentence + '\n')
        
        
def get_Prime_number_in_a_range(start, end):
    Prime_list = []
    for val in range(start, end + 1): 
        prime_or_not = True
        for n in range(2, val):
            if (val % n) == 0:
                prime_or_not = False
                break
        if prime_or_not:
            Prime_list.append(val)
    return Prime_list


def get_out_channel_number(paramenter_layer, in_channel, prime_list):
    out_channel_expect = int(paramenter_layer/(in_channel*sum(prime_list)))
    return out_channel_expect

def generate_layer_parameter_list(start,end,paramenter_number_of_layer_list, in_channel = 1):
    prime_list = get_Prime_number_in_a_range(start, end)
    if prime_list == []:
        print('start = ',start, 'which is larger than end = ', end)
    paramenter_number_of_layer_list[0] =  paramenter_number_of_layer_list[0]*in_channel
    input_in_channel = in_channel
    layer_parameter_list = []
    for paramenter_number_of_layer in paramenter_number_of_layer_list:
        out_channel = get_out_channel_number(paramenter_number_of_layer, in_channel, prime_list)
        
        tuples_in_layer= []
        for prime in prime_list:
            tuples_in_layer.append((in_channel,out_channel,prime))
        in_channel =  len(prime_list)*out_channel
        
        layer_parameter_list.append(tuples_in_layer)
    
    tuples_in_layer_last = []
    first_out_channel = len(prime_list)*get_out_channel_number(paramenter_number_of_layer_list[0], input_in_channel, prime_list)
    tuples_in_layer_last.append((in_channel,first_out_channel,start))
    tuples_in_layer_last.append((in_channel,first_out_channel,start+1))
    layer_parameter_list.append(tuples_in_layer_last)
    return layer_parameter_list


def calculate_mask_index(kernel_length_now,largest_kernel_lenght):
    right_zero_mast_length = math.ceil((largest_kernel_lenght-1)/2)-math.ceil((kernel_length_now-1)/2)
    left_zero_mask_length = largest_kernel_lenght - kernel_length_now - right_zero_mast_length
    return left_zero_mask_length, left_zero_mask_length+ kernel_length_now

def creat_mask(number_of_input_channel,number_of_output_channel, kernel_length_now, largest_kernel_lenght):
    ind_left, ind_right= calculate_mask_index(kernel_length_now,largest_kernel_lenght)
    mask = np.ones((number_of_input_channel,number_of_output_channel,largest_kernel_lenght))
    mask[:,:,0:ind_left]=0
    mask[:,:,ind_right:]=0
    return mask


def creak_layer_mask(layer_parameter_list):
    largest_kernel_lenght = layer_parameter_list[-1][-1]
    mask_list = []
    init_weight_list = []
    bias_list = []
    for i in layer_parameter_list:
        conv = torch.nn.Conv1d(in_channels=i[0], out_channels=i[1], kernel_size=i[2])
        ind_l,ind_r= calculate_mask_index(i[2],largest_kernel_lenght)
        big_weight = np.zeros((i[1],i[0],largest_kernel_lenght))
        big_weight[:,:,ind_l:ind_r]= conv.weight.detach().numpy()
        
        bias_list.append(conv.bias.detach().numpy())
        init_weight_list.append(big_weight)
        
        mask = creat_mask(i[1],i[0],i[2], largest_kernel_lenght)
        mask_list.append(mask)
        
    mask = np.concatenate(mask_list, axis=0)
    init_weight = np.concatenate(init_weight_list, axis=0)
    init_bias = np.concatenate(bias_list, axis=0)
    return mask.astype(np.float32), init_weight.astype(np.float32), init_bias.astype(np.float32)

    
class build_layer_with_layer_parameter(nn.Module):
    def __init__(self,layer_parameters):
        super(build_layer_with_layer_parameter, self).__init__()

        os_mask, init_weight, init_bias= creak_layer_mask(layer_parameters)
        
        
        in_channels = os_mask.shape[1] 
        out_channels = os_mask.shape[0] 
        max_kernel_size = os_mask.shape[-1]

        self.weight_mask = nn.Parameter(torch.from_numpy(os_mask),requires_grad=False)
        
        self.padding = nn.ConstantPad1d((int((max_kernel_size-1)/2), int(max_kernel_size/2)), 0)
         
        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=max_kernel_size)
        self.conv1d.weight = nn.Parameter(torch.from_numpy(init_weight),requires_grad=True)
        self.conv1d.bias =  nn.Parameter(torch.from_numpy(init_bias),requires_grad=True)

        self.bn = nn.BatchNorm1d(num_features=out_channels)
    
    def forward(self, X):
        self.conv1d.weight.data = self.conv1d.weight*self.weight_mask
        #self.conv1d.weight.data.mul_(self.weight_mask)
        result_1 = self.padding(X)
        result_2 = self.conv1d(result_1)
        result_3 = self.bn(result_2)
        result = F.relu(result_3)
        return result    
    
    
class OS_CNN_block(nn.Module):
    def __init__(self,layer_parameter_list,n_class,squeeze_layer = True):
        super(OS_CNN_block, self).__init__()
        self.squeeze_layer = squeeze_layer
        self.layer_parameter_list = layer_parameter_list
        self.layer_list = []
        
        
        for i in range(len(layer_parameter_list)):
            layer = build_layer_with_layer_parameter(layer_parameter_list[i])
            self.layer_list.append(layer)
        
        self.net = nn.Sequential(*self.layer_list)
            
        self.averagepool = nn.AdaptiveAvgPool1d(1)
        
        out_put_channel_numebr = 0
        for final_layer_parameters in layer_parameter_list[-1]:
            out_put_channel_numebr = out_put_channel_numebr+ final_layer_parameters[1] 
            
        self.hidden = nn.Linear(out_put_channel_numebr, n_class)

    def forward(self, X):
        
        X = self.net(X)
        
        if self.squeeze_layer:
            X = self.averagepool(X)
            X = X.squeeze_(-1)
            X = self.hidden(X)
        return X

def check_channel_limit(os_block_layer_parameter_list,n_input_channel,mid_channel_limit): 
    out_channel_each = 0
    for conv_in in os_block_layer_parameter_list[-1]:
        out_channel_each = out_channel_each + conv_in[1]
    total_temp_channel = n_input_channel*out_channel_each
    if total_temp_channel<=mid_channel_limit:
        return os_block_layer_parameter_list
    else:
        
        temp_channel_each = max(int(mid_channel_limit/(n_input_channel*len(os_block_layer_parameter_list[-1]))),1)
        for i in range(len(os_block_layer_parameter_list[-1])):
            os_block_layer_parameter_list[-1][i]= (os_block_layer_parameter_list[-1][i][0],
                                                   temp_channel_each,
                                                  os_block_layer_parameter_list[-1][i][2])
        print('reshape temp channel from ',total_temp_channel,' to ',n_input_channel,' * ',temp_channel_each,)
        return os_block_layer_parameter_list

    
class OS_CNN(nn.Module):
    def __init__(self, layer_parameter_list, n_class, n_input_channel,squeeze_layer = True):
        super(OS_CNN, self).__init__()
        
        self.mid_channel_limit = 1000
        self.squeeze_layer = squeeze_layer
        self.layer_parameter_list = layer_parameter_list
        self.OS_block_list = nn.ModuleList()
        
        os_block_layer_parameter_list = copy.deepcopy(layer_parameter_list[:-1])
        os_block_layer_parameter_list = check_channel_limit(os_block_layer_parameter_list,n_input_channel,self.mid_channel_limit)
        print('os_block_layer_parameter_list is     :',os_block_layer_parameter_list)
        for nth in range(n_input_channel):
            torch_OS_CNN_block = OS_CNN_block(os_block_layer_parameter_list,n_class, False)
            self.OS_block_list.append(torch_OS_CNN_block)
        
        rf_size = layer_parameter_list[0][-1][-1]
        in_channel_we_want= len(layer_parameter_list[1])*os_block_layer_parameter_list[-1][-1][1]*n_input_channel
        print('in_channel_we_want is           :', in_channel_we_want)
       
        layer_parameter_list = generate_layer_parameter_list(1,rf_size+1,[8*128, (5*128*256 + 2*256*128)/2],in_channel = in_channel_we_want)
        
        self.averagepool = nn.AdaptiveAvgPool1d(1) 
        print('layer_parameter_list:    ', layer_parameter_list)
        self.OS_net =  OS_CNN_block(layer_parameter_list,n_class, True)

    def forward(self, X):
        OS_block_result_list = []
        for i_th_channel, OS_block in enumerate(self.OS_block_list):
            OS_block_result = OS_block(X[:,i_th_channel:i_th_channel+1,:])
            OS_block_result_list.append(OS_block_result)
        result = F.relu(torch.cat(tuple(OS_block_result_list), 1)) 
        
        result = self.OS_net(result)
        return result

class OS_CNN_easy_use():
    
    def __init__(self,
                 log_dir,
                 model_dir, 
                 dataset_name,
                 nsteps, 
                 device, 
                 Max_kernel_size = 89, 
                 paramenter_number_of_layer_list = [8*128, 5*128*256 + 2*256*128], 
                 max_epoch = 100, 
                 batch_size=256,
                 print_result_every_x_epoch = 10,
                 start_kernel_size = 1,
                 quarter_or_half = 4
                ):
        
        super(OS_CNN_easy_use, self).__init__()
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_save_path = model_dir + dataset_name+ '_' + str(nsteps) + '.pth'
        

        self.Result_log_folder = log_dir
        self.dataset_name = dataset_name        
        self.model_save_path = model_save_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        
        self.Max_kernel_size = Max_kernel_size
        self.paramenter_number_of_layer_list = paramenter_number_of_layer_list
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.print_result_every_x_epoch = print_result_every_x_epoch
        self.quarter_or_half = quarter_or_half
        
        self.OS_CNN = None
        
        
        self.start_kernel_size = start_kernel_size
        
    def fit(self, X_train, y_train, X_val, y_val):

        print('code is running on ',self.device)
        
        # covert numpy to pytorch tensor and put into gpu
        X_train = torch.from_numpy(X_train).float()
        X_train.requires_grad = False
        if len(X_train.shape) ==3:
            X_train = X_train.to(self.device)
        else:
            X_train = X_train.unsqueeze_(1).to(self.device)
        y_train = torch.from_numpy(y_train).to(self.device)
        n_input_channel = X_train.shape[1]
        
        X_test = torch.from_numpy(X_val).float()
        X_test.requires_grad = False
        if len(X_test.shape) ==3:
            X_test = X_test.to(self.device)
        else:
            X_test = X_test.unsqueeze_(1).to(self.device)
        y_test = torch.from_numpy(y_val).to(self.device)
        
        
        input_shape = X_train.shape[-1]
        n_class = max(y_train) + 1
        receptive_field_shape= min(int(X_train.shape[-1]/self.quarter_or_half),self.Max_kernel_size)
        
        # generate parameter list
        layer_parameter_list = generate_layer_parameter_list(self.start_kernel_size,receptive_field_shape,self.paramenter_number_of_layer_list,in_channel = 1)
        torch_OS_CNN = OS_CNN(layer_parameter_list, n_class.item(),n_input_channel, True).to(self.device)
        
        # save_initial_weight
        # torch.save(torch_OS_CNN.state_dict(), self.Initial_model_path)
        
        
        # loss, optimizer, scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(torch_OS_CNN.parameters())
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=200, min_lr=0.0001)
        
        # build dataloader
        print('batch_size:  ', max(int(min(X_train.shape[0] / 10, self.batch_size)),2))
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=max(int(min(X_train.shape[0] / 10, self.batch_size)),2), shuffle=True)
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=max(int(min(X_train.shape[0] / 10, self.batch_size)),2), shuffle=False)
        
        
        torch_OS_CNN.train()   
        
        for i in range(self.max_epoch):
            for sample in train_loader:
                optimizer.zero_grad()
                y_predict = torch_OS_CNN(sample[0])
                output = criterion(y_predict, sample[1])
                output.backward()
                optimizer.step()
            scheduler.step(output)
            
            if eval_condition(i,self.print_result_every_x_epoch):
                for param_group in optimizer.param_groups:
                    print('epoch =',i, 'lr = ', param_group['lr'])
                torch_OS_CNN.eval()
                acc_train = eval_model(torch_OS_CNN, train_loader)
                acc_test = eval_model(torch_OS_CNN, test_loader)
                torch_OS_CNN.train()
                print('train_acc=\t', acc_train, '\t test_acc=\t', acc_test, '\t loss=\t', output.item())
                sentence = 'train_acc=\t'+str(acc_train)+ '\t test_acc=\t'+str(acc_test) 
                print('log saved at:')
                save_to_log(sentence,self.Result_log_folder, self.dataset_name)
                torch.save(torch_OS_CNN.state_dict(), self.model_save_path)
         
        torch.save(torch_OS_CNN.state_dict(), self.model_save_path)
        self.OS_CNN = torch_OS_CNN

        
        
    def predict(self, X_test):
        
        X_test = torch.from_numpy(X_test).float()
        X_test.requires_grad = False
        if len(X_test.shape) ==3:
            X_test = X_test.to(self.device)
        else:
            X_test = X_test.unsqueeze_(1).to(self.device)
        
        test_dataset = TensorDataset(X_test)
        test_loader = DataLoader(test_dataset, batch_size=int(min(X_test.shape[0] / 10, self.batch_size)), shuffle=False)
        
        self.OS_CNN.eval()
        
        predict_list = np.array([])
        for sample in test_loader:
            y_predict = self.OS_CNN(sample[0])
            y_predict = y_predict.detach().cpu().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            
        return predict_list
    
    def one_predict(self, X_test):
        
        X_test = torch.from_numpy(X_test).float()
        X_test.requires_grad = False
        if len(X_test.shape) ==3:
            X_test = X_test.to(self.device)
        else:
            X_test = X_test.unsqueeze_(1).to(self.device)
        
        self.OS_CNN.eval()
        
        y_predict = self.OS_CNN(X_test)
        y_predict = y_predict.detach().cpu().numpy()
        
        return y_predict

def max_pooling_data(a,sampling):
    a = np.float32(a)
    a_torch = torch.from_numpy(a)
    a_torch.unsqueeze_(1)
    m = nn.MaxPool1d(sampling, stride=sampling)
    output = m(a_torch)
    return output.squeeze_(1).numpy()

def down_sampling_data(a,sampling):
    return a[:,::sampling]

def long_data_to_more_channels(a, channels):
    more_length_required = channels - a.shape[-1]%channels
    padding_size = list(a.shape)
    padding_size[-1]= int(more_length_required)
    padding = np.zeros(tuple(padding_size))
    a = np.concatenate((a, padding), axis=-1)
    result = 1
    for i in a.shape[1:]:
        result = result*i
    result = result*channels/a.shape[-1]
    z = np.swapaxes(np.reshape(a,(a.shape[0],int(a.shape[-1]/channels),int(result))),1,2)
    return np.float32(z)    

def prepare_agent(env_name, input_tag=False):
          
    if env_name == 'BipedalWalkerHC':
        env = gym.make('BipedalWalker-v3',
                       hardcore=True,)
        model = SAC.load('gymmodel/BipedalWalkerHC.zip')
        if input_tag:
            num_nodes = 28
        else:
            num_nodes = 24
        
        alg_tag = 'SAC'

    elif env_name == 'Walker2d':
        env = gym.make('Walker2d-v4')
        model = SAC.load('gymmodel/Walker2d.zip')
        if input_tag:
            num_nodes = 23
        else:
            num_nodes = 17
        
        alg_tag = 'SAC'
    elif env_name == 'InvertedDoublePendulum':
        env = gym.make('InvertedDoublePendulum-v4')
        model = PPO.load('gymmodel/InvertedDoublePendulum.zip')
        if input_tag:
            num_nodes = 12
        else:
            num_nodes = 11
        
        alg_tag = 'PPO'
    
    elif env_name == 'Hopper':
        env = gym.make('Hopper-v4')
        model = SAC.load('gymmodel/Hopper.zip')
        if input_tag:
            num_nodes = 14
        else:
            num_nodes = 11
    
        alg_tag = 'SAC'
    
    elif env_name == 'Humanoid':
        env = gym.make('Humanoid-v4')
        model = SAC.load('gymmodel/Humanoid.zip')
        if input_tag:
            num_nodes = 62
        else:
            num_nodes = 45
        
        alg_tag = 'SAC'
        
    return env, model, num_nodes, alg_tag

def transform_input(obs, action, model, record, alg_tag):
    # SAC version
    state = torch.as_tensor(obs).unsqueeze(0).to(model.device)
    actions = torch.as_tensor(action).unsqueeze(0).to(model.device)
    record.append(torch.cat([state[:,:45], actions], dim=1))

    return record
