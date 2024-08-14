import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import os



def validation(model, testloader, criterion, device='cpu'):
    accuracy = 0
    test_loss = 0
    num = 0
    for inputs, labels in testloader:
        inputs = inputs.float()
        inputs, labels = inputs.to(device), labels.to(device)

        batch_size = inputs.shape[0]
        seql = inputs.shape[1]
        seq_lens = torch.tensor([seql] * batch_size, dtype=torch.float)
        output = model.forward(inputs, seq_lens)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy


def train(model, trainloader, validloader, criterion, optimizer, 
          epochs=10, print_every=10, device='cpu', run_name='model_mlstm_fcn'):
    print("Training started on device: {}".format(device))

    save_dir = './model/MLSTM-FCN'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    valid_loss_min = np.Inf # track change in validation loss
    steps = 0
    
    for e in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for inputs, labels in trainloader:
            steps += 1

            inputs = inputs.float()
            inputs, labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            batch_size = inputs.shape[0]
            seql = inputs.shape[1]
            seq_lens = torch.tensor([seql] * batch_size, dtype=torch.float)
            output = model.forward(inputs, seq_lens)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, device)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.6f}.. ".format(train_loss/print_every),
                      "Val Loss: {:.6f}.. ".format(valid_loss/len(validloader)),
                      "Val Accuracy: {:.2f}%".format(accuracy/len(validloader)*100))
                
                # save model if validation loss has decreased
                if valid_loss <= valid_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
                    torch.save(model.state_dict(), './model/'+run_name+'.pth')
                    valid_loss_min = valid_loss

                train_loss = 0

                model.train()



def load_datasets(dataset_name='BipedalWalkerHC', nsteps=20):
    dataset_name = dataset_name + '_' + str(nsteps)
    data_path = './data/Train/' + dataset_name + '/'

    X_train = torch.load(data_path+'X_train.pt').squeeze(1).transpose(1, 2)
    X_valid = torch.load(data_path+'X_valid.pt').squeeze(1).transpose(1, 2)

    y_train = torch.load(data_path+'y_train.pt')
    y_valid = torch.load(data_path+'y_valid.pt')


    print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)

    return train_dataset, val_dataset
