import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
from utils_weighted import *
from utils import get_loss_acc
from model import transformer_base, transformer_large, transformer_huge
from score import get_high_quali_pred

from utils_weighted import FEATURE_ARCHIVE


class mydataset_w_weight(Dataset):
    def __init__(self, X, Y, Weights):
        self.Data = torch.Tensor(X)
        self.Label = torch.LongTensor(Y)
        self.weights = torch.Tensor(Weights)

    def __getitem__(self, item):
        return self.Data[item], self.Label[item], self.weights[item]

    def __len__(self):
        return len(self.Label)

# initialization weights uniformly
def prepare_data_w_weight(test_ratio = 0.2, weights = np.ones(951) / 951.0, seed = 20220712):
    all_data = np.load(FEATURE_ARCHIVE + "all_feature_interp951.npz", allow_pickle=True)
    X_all = all_data["X"]
    Y_all = all_data["Y"]
    X_all = X_all.reshape(-1, 100, 34).transpose(0, 2, 1)

    # split data
    np.random.seed(seed)
    n_samples = X_all.shape[0]
    assert n_samples == len(weights), "The number of weights({}) doesn't match the number of samples({})".format(len(weights), n_samples)
    test_size = int(n_samples * test_ratio)
    perm = np.random.permutation(n_samples)
    test_idx = perm[:test_size]
    train_idx = perm[test_size:]
    X_train, Y_train, weights_train = X_all[train_idx], Y_all[train_idx], weights[train_idx]
    X_test, Y_test, weights_test = X_all[test_idx], Y_all[test_idx], weights[test_idx]

    # create dataset and dataloader
    train_dataset = mydataset_w_weight(X_train, Y_train, weights_train)
    test_dataset = mydataset_w_weight(X_test, Y_test, weights_test)
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    return trainloader, testloader, test_idx


def train_w_weight(model, epochs, trainloader, testloader, optimizer, criterion, save_path):
    """
    Train a neural network and save the best model on accoding to its performance on testloader
    :param model: the model to be trained
    :param epochs: number of epochs for training
    :param weight: confidence scores work as weight in training and testing
    :param trainloader: train dataloader
    :param testloader: test dataloader
    :param optimizer: the optimizer for gradient descent
    :param criterion: the loss function
    :param save_path: the path for saving model parameters
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Trained on {}".format(device))
    model.to(device)

    # train model
    best_test_acc = 0
    model.train()
    for epoch in range(epochs):
        for batch in trainloader:
            X_batch, Y_batch = batch[0].to(device), batch[1].to(device)
            if len(batch) == 3:
                 weight_batch = batch[2].to(device)
            # forward
            logits = model(X_batch)
            probs = torch.softmax(logits, dim = 1)
            loss = torch.mul(criterion(logits, Y_batch), weight_batch)
            loss_reducted = torch.mean(loss)
            # backward
            optimizer.zero_grad()
            loss_reducted.backward()
            optimizer.step()

        # evaluate
        with torch.no_grad():
            model.eval()
            # evaluate train
            train_loss, train_acc = get_loss_acc_w_weight(model, trainloader, criterion) # weighted training
            # evaluate test
            test_loss, test_acc = get_loss_acc(model, testloader) # test metrics are not weighted! -> default criterion

        # save model weights if it's the best
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print("Epoch {:>4}/{:>4} | Training loss: {:>8} | Testing loss: {:>8} | Training acc: {:>8} | Testing acc: {:>8}".format(
                epoch+1, epochs, train_loss, test_loss, train_acc, test_acc
            ))
            print("Saved!")
    print("Epoch {:>4}/{:>4} | Training loss: {:>8} | Testing loss: {:>8} | Training acc: {:>8} | Testing acc: {:>8}".format(
        epoch+1, epochs, train_loss, test_loss, train_acc, test_acc
    ))


def train_transfomer_w_weight(size="base"):
    """
    Train a transformer model 5 times with different train test split
    :param size: Size of the transformer model. "base" or "large" or "huge"
    """
    seed = 20220728
    model_save_dir = "./model_weights/"
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    for i in range(5):
        this_seed = seed + i
        torch.manual_seed(this_seed)
        save_path = model_save_dir + "Transformer_{}_{}_weighted.pth".format(size, i+1)
        trainloader, testloader, test_idx = prepare_data_w_weight(test_ratio=0.2, seed=this_seed)

        # train model
        if size == "base":
            model = transformer_base()
        elif size == "large":
            model = transformer_large()
        elif size == "huge":
            model = transformer_huge()

        epochs = 200
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(reduction = 'none')
        train_w_weight(model, epochs, trainloader, testloader, optimizer, criterion, save_path)


def train_transformer_EM(iterations = 3, method = "attn", size = "huge"):
    """
    Train a transformer model 5 times with different train test split. 
    For each model, update sample weights by confidence score and retrain the model for weighted loss
    :iterations: number of iterations for EM training
    :method: the method to compute sample weights
    :size: Size of the transformer model. "base" or "large" or "huge"
    """
    seed = 20220728
    model_save_dir = "../model_weights/"
    result_save_dir = "../results/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    
    for i in range(5):
        this_seed = seed + i
        torch.manual_seed(this_seed) # reproducible
        # instantiate model
        if size == "base":
            model = transformer_base()
        elif size == "large":
            model = transformer_large()
        elif size == "huge":
            model = transformer_huge()
        
        # iterative training
        for iter in range(iterations):
            print('-' * 60)
            print(f'Size: {size} | Seed: {this_seed} | Iteration: {iter:4d} | Method: {method} ')
            save_path = model_save_dir + "Transformer_{}_seed_{}_weighted_{}_iter_{}.pth".format(size, i+1, method, iter)

            if iter == 0:
                weights = np.ones(951) / 951.0 # initialize weights -> unweighted at first
            trainloader, testloader, test_idx = prepare_data_w_weight(test_ratio = 0.2, seed = this_seed, weights = weights)

            epochs = 200
            optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
            criterion = nn.CrossEntropyLoss(reduction = 'none')
            train_w_weight(model, epochs, trainloader, testloader, optimizer, criterion, save_path) # weights inside of dataloaders
            weights = get_conf(model = model, seed = this_seed, method = method) # update weights

            # evaluation
            with torch.no_grad():
                # get the best model
                if size == "base":
                    model = transformer_base()
                elif size == "large":
                    model = transformer_large()
                elif size == "huge":
                    model = transformer_huge()
                model.to(device)
                model.load_state_dict(torch.load(save_path, map_location = device))

                accuracy_arr = get_high_quali_pred(model = model, seed = this_seed, method = method)
                result_save_path = result_save_dir + "accu_Transformer_{}_seed_{}_weighted_{}_iter_{}.npy".format(size, i+1, method, iter)
                np.save(arr = accuracy_arr, file = result_save_path)
                print(f'Accuracy Array: {accuracy_arr}')
    pass


if __name__ == "__main__":
    result_save_dir = "../results/"
    if not os.path.exists(result_save_dir):
        os.mkdir(result_save_dir)

    for name in ["base", "large", "huge"]: # for each structure
        train_transfomer_w_weight(size = name)

