import torch
import torch.nn as nn
import torch.optim as optim

from src.model import MLSTMfcn
from src.utils import train, load_datasets
from src.constants import NUM_CLASSES, MAX_SEQ_LEN, NUM_FEATURES
import warnings

warnings.filterwarnings("ignore")

def main():
    dataset = args.dataset
    n = args.nsteps
    run_name = 'MLSTM-FCN/' + args.dataset + '_' + str(n)

    assert dataset in NUM_CLASSES.keys()

    # train_dataset, val_dataset, _ = load_datasets(dataset_name=dataset)
    train_dataset, val_dataset = load_datasets(dataset_name=dataset, nsteps=n)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: {}".format(device))

    print("NUM_CLASSES: ", NUM_CLASSES[dataset], "MAX_SEQ_LEN: ", MAX_SEQ_LEN[dataset], "NUM_FEATURES: ", NUM_FEATURES[dataset])

    mlstm_fcn_model = MLSTMfcn(num_classes=NUM_CLASSES[dataset], 
                               max_seq_len=MAX_SEQ_LEN[dataset], 
                               num_features=NUM_FEATURES[dataset])
    # mlstm_fcn_model = MLSTMfcn(num_classes=2, 
    #                            max_seq_len=20,
    #                            num_features=24)
    mlstm_fcn_model.to(device)

    optimizer = optim.SGD(mlstm_fcn_model.parameters(), lr=args.learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    train(mlstm_fcn_model, 
          train_loader, 
          val_loader, 
          criterion, 
          optimizer, 
          epochs=args.epochs, 
          print_every=10, 
          device=device, 
          run_name=run_name)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--name", type=str, default="model_mlstm_fcn")
    p.add_argument("--dataset", type=str, default="ISLD")
    p.add_argument('-n', '--nsteps', type=int, default=20)
    args = p.parse_args()
    main()
