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

def prepare_data(test_ratio=0.2, val_ratio = 0.1, seed=20220712):
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
    val_size = int(n_samples * val_ratio)
    perm = np.random.permutation(n_samples)
    test_idx = perm[:test_size]
    val_idx = perm[test_size:(test_size + val_size)]
    train_idx = perm[(test_size + val_size):]
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    Y_train, Y_test = Y_all[train_idx], Y_all[test_idx]
    X_val, Y_val = X_all[val_idx], Y_all[val_idx]

    # create dataset and dataloader
    train_dataset = mydataset(X_train, Y_train)
    test_dataset = mydataset(X_test, Y_test)
    val_dataset = mydataset_w_weight(X_val, Y_val, weights_val)
    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    valloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    return trainloader, testloader

# initialization weights uniformly
def prepare_data_w_weight(test_ratio = 0.2, val_ratio = 0.1, weights = np.ones(951), seed = 20220712):
    all_data = np.load(FEATURE_ARCHIVE + "all_feature_interp951.npz", allow_pickle=True)
    X_all = all_data["X"]
    Y_all = all_data["Y"]
    X_all = X_all.reshape(-1, 100, 34).transpose(0, 2, 1)

    # split data
    np.random.seed(seed)
    n_samples = X_all.shape[0]
    assert n_samples == len(weights), "The number of weights({}) doesn't match the number of samples({})".format(len(weights), n_samples)
    test_size = int(n_samples * test_ratio)
    val_size = int(n_samples * val_ratio)
    perm = np.random.permutation(n_samples)
    test_idx = perm[:test_size]
    val_idx = perm[test_size:(test_size + val_size)]
    train_idx = perm[(test_size + val_size):]
    X_train, Y_train, weights_train = X_all[train_idx], Y_all[train_idx], weights[train_idx]
    X_test, Y_test, weights_test = X_all[test_idx], Y_all[test_idx], weights[test_idx]
    X_val, Y_val, weights_val = X_all[val_idx], Y_all[val_idx], weights[val_idx]

    # create dataset and dataloader
    train_dataset = mydataset_w_weight(X_train, Y_train, weights_train)
    test_dataset = mydataset_w_weight(X_test, Y_test, weights_test)
    val_dataset = mydataset_w_weight(X_val, Y_val, weights_val)
    trainloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    valloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    return trainloader, valloader, testloader, test_idx

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


def get_conf(model, seed, method = "attn"):
    '''
    Compute confidence score based on attention for all video samples
    :model: trained transformer model
    :method: attn / margin / entropy
    :output: confidence score for each sample
    '''

    CONF_DIR = FEATURE_ARCHIVE + "confidence_scores_by_frame.npy"
    conf_keypoints = np.load(CONF_DIR) # raw confidence score in shape (n_sapmles, n_frames, n_keypoints)
    conf_keypoints = np.mean(conf_keypoints, axis = 2) # (n_sapmles, n_frames)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    frame_attn_list = []
    prob_list = []
    model.to(device)

    all_data = np.load(FEATURE_ARCHIVE + "all_feature_interp951.npz", allow_pickle=True)
    X_all = all_data["X"]
    Y_all = all_data["Y"]
    X_all = X_all.reshape(-1, 100, 34).transpose(0, 2, 1)
    all_dataset = mydataset(X_all, Y_all)
    allloader = DataLoader(all_dataset, batch_size = 64, shuffle = False)

    if method == "attn":
        with torch.no_grad():
            # mask = np.ones(conf_keypoints.shape[0], dtype = np.bool)
            # mask[test_idx] = 0
            # conf_keypoints_train = conf_keypoints[mask, ...] # only for training data
            for batch in allloader:
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

            frame_attn = np.concatenate(frame_attn_list, axis = 0) # (n_samples, length, length)
            frame_attn = np.mean(frame_attn, axis = 2) # (n_samples, length) -> attention by frame
            # elementwise product
            conf_score = conf_keypoints * frame_attn # elementwise product -> (n_samples, n_frames)
            conf_score = np.mean(conf_score, axis = 1) # (n_samples, )
            # minmax scalar
            conf_score = (conf_score - np.min(conf_score)) / (np.max(conf_score) - np.min(conf_score))
            # return conf_score, frame_attn
    elif method == "margin":
        with torch.no_grad():
            for batch in allloader:
                X_batch, Y_batch = batch[0].to(device), batch[1].to(device)
                if len(batch) == 3:
                    weight_batch = batch[2].to(device)
                # forward
                logits = model(X_batch)
                probs = torch.softmax(logits, dim = 1).cpu().numpy()
                prob_list.append(probs)
            
            pred = np.concatenate(prob_list, axis = 0)
            conf_score = np.empty(pred.shape[0])

            for i in range(pred.shape[0]):
                temp_max_prob = np.max(pred[i, :])
                temp_second_highest_prob = np.partition(pred[i, :].flatten(), -2)[-2]
                conf_score[i] = np.abs(temp_max_prob - temp_second_highest_prob)
    elif method == "entropy":
        with torch.no_grad():
            for batch in allloader:
                X_batch, Y_batch = batch[0].to(device), batch[1].to(device)
                if len(batch) == 3:
                    weight_batch = batch[2].to(device)
                # forward
                logits = model(X_batch)
                probs = torch.softmax(logits, dim = 1).cpu().numpy()
                prob_list.append(probs)
            
            pred = np.concatenate(prob_list, axis = 0)
            conf_score = np.empty(pred.shape[0])

            for i in range(pred.shape[0]):
                log_prob = np.log2(pred[i, :])
                entropy_vec = np.multiply(log_prob, pred[i,:])
                conf_score[i] = -np.sum(entropy_vec)

    # # minmax scalar
    # conf_score_std = (conf_score - np.min(conf_score)) / (np.max(conf_score) - np.min(conf_score))

    return conf_score


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
