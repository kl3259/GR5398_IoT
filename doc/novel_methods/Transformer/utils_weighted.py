import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import numpy as np


FEATURE_ARCHIVE = "../../feature_archive/"


class mydataset(Dataset):
    def __init__(self, X, Y):
        self.Data = torch.Tensor(X)
        self.Label = torch.LongTensor(Y)

    def __getitem__(self, item):
        return self.Data[item], self.Label[item]

    def __len__(self):
        return len(self.Label)

class mydataset_w_weight(Dataset):
    def __init__(self, X, Y, Weights):
        self.Data = torch.Tensor(X)
        self.Label = torch.LongTensor(Y)
        self.weights = torch.Tensor(Weights)

    def __getitem__(self, item):
        return self.Data[item], self.Label[item], self.weights[item]

    def __len__(self):
        return len(self.Label)

def prepare_data(test_ratio=0.2, seed=20220712):
    """
    Prepare two dataloader for train and test. The dataset contains 951 samples.
    :param test_ratio: Proportion of test data
    :param seed: random seed for train test split
    :return: trainloader, testloader
    """
    all_data = np.load(FEATURE_ARCHIVE + "all_feature_interp951.npz")
    X_all = all_data["X"]
    Y_all = all_data["Y"]

    # reshape data to (n_samples, n_features, length)
    X_all = X_all.reshape(-1, 100, 34).transpose(0, 2, 1)

    # train test split
    np.random.seed(seed)
    n_samples = X_all.shape[0]
    test_size = int(n_samples * test_ratio)
    perm = np.random.permutation(n_samples)
    test_idx = perm[:test_size]
    train_idx = perm[test_size:]
    X_train = X_all[train_idx]
    Y_train = Y_all[train_idx]
    X_test = X_all[test_idx]
    Y_test = Y_all[test_idx]

    # create dataset and dataloader
    train_dataset = mydataset(X_train, Y_train)
    test_dataset = mydataset(X_test, Y_test)
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    return trainloader, testloader

# initialization weights uniformly
def prepare_data_w_weight(test_ratio=0.2, weights = np.ones(951) / 951.0, seed = 20220712):
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

# def get_loss_acc(model, dataloader, criterion=nn.CrossEntropyLoss()):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     correct = 0
#     total = 0
#     total_loss = 0
#     num_batches = 0

#     with torch.no_grad():
#         model.eval()
#         for batch in dataloader:
#             X_batch, Y_batch = batch[0].to(device), batch[1].to(device)
#             if len(batch) == 3:
#                 index_batch = batch[2]   # in case we need the index of samples in this batch
#             total += len(Y_batch)
#             num_batches += 1
#             logits = model(X_batch)
#             y_pred = torch.argmax(logits, dim=1)
#             correct += torch.sum(y_pred == Y_batch).cpu().numpy()
#             loss = criterion(logits, Y_batch)
#             total_loss += loss.item()
#     acc = correct / total
#     total_loss = total_loss / num_batches

#     return total_loss, acc


def get_conf(model, seed):
    '''
    Compute confidence score based on attention for each training data
    :model: trained transformer model
    :output: confidence score for each sample
    '''
    CONF_DIR = FEATURE_ARCHIVE + "confidence_scores_by_frame.npy"
    conf_keypoints = np.load(CONF_DIR) # raw confidence score in shape (n_videos, n_frame, n_keypoints)
    conf_keypoints = np.mean(conf_keypoints, axis = 2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    frame_attn_list = []
    model.to(device)
    with torch.no_grad():
        trainloader, _, test_idx = prepare_data_w_weight(seed = seed)
        mask = np.ones(conf_keypoints.shape[0], dtype = np.bool)
        mask[test_idx] = 0
        conf_keypoints_train = conf_keypoints[mask, ...] # only for training data
        for batch in trainloader:
            X_batch, Y_batch = batch[0].to(device), batch[1].to(device)
            if len(batch) == 3:
                 weight_batch = batch[2].to(device)
            # forward
            logits = model(X_batch)
            probs = torch.softmax(logits, dim = 1)
            frame_attn = model.blocks[-1].attn.attn # (B, n_heads, length + 1, length + 1)
            frame_attn = frame_attn[:,:,:100, :100] # remove cls token and positional encoding
            frame_attn_single_head = torch.mean(frame_attn, dim = 1).cpu().numpy() # (B, length, length)
            frame_attn_list.append(frame_attn_single_head)
        frame_attn_train = np.concatenate(frame_attn_list, axis = 0) # (n_training, length, length)
        frame_attn_train = np.mean(frame_attn_train, axis = 2) # (n_training, length) -> attention by frame
        # elementwise product
        conf_score = conf_keypoints_train * frame_attn_train # elementwise product -> (n_training, n_frames)
        conf_score = np.mean(conf_score, axis = 1) # (n_training, )
        # minmax scalar
        conf_score_std = (conf_score - np.min(conf_score)) / (np.max(conf_score) - np.min(conf_score))

    return conf_score_std


def get_loss_acc_w_weight(model, dataloader, criterion = nn.CrossEntropyLoss(reduction = 'none')):
    '''
    Comupute weighed loss function and accuracy based on weights batch
    :model: transformer model
    :dataloader: dataloader with x and y in batches
    :criterion: loss function
    :return: weighted total_loss and weighted accuracy
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    correct = 0.0
    total = 0.0
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        model.eval()
        for batch in dataloader:
            X_batch, Y_batch = batch[0].to(device), batch[1].to(device)
            if len(batch) == 3:
                weight_batch = batch[2].to(device)   # in case we need the index of samples in this batch
            total += torch.sum(weight_batch)
            num_batches += 1
            logits = model(X_batch)
            probs = torch.softmax(logits, dim = 1)
            y_pred = torch.argmax(logits, dim = 1)
            count_correct = (y_pred == Y_batch).to(device)
            correct += torch.sum(torch.mul(weight_batch, count_correct)).cpu().numpy()
            loss = criterion(probs, Y_batch)
            weighted_loss = torch.mean(torch.mul(loss, weight_batch)) # modify
            total_loss += weighted_loss.item()
    acc = correct / total # weighted acc
    total_loss = total_loss / num_batches

    return total_loss, acc


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    count = sum(p.numel() for p in model.parameters())
    print("Total parameters: {}".format(count))
    print("Trainable parameters: {}".format(trainable))
    return count, trainable

if __name__ == "__main__":
    pass
