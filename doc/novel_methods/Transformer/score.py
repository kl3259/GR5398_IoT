import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from model import transformer_base, transformer_huge, transformer_large
from train_weighted import prepare_data_w_weight
from utils_weighted import FEATURE_ARCHIVE

DETECTRON_SCORE_PATH = FEATURE_ARCHIVE + 'confidence_scores_by_frame.npy'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "mps"

def init_confidence_score():
    '''
    Initialize video confidence scores based on attention by averaging all the keypoints' confidence score. 
    :return: trainloader, testloader
    '''
    scores = np.load(DETECTRON_SCORE_PATH)
    scores_all = scores
    scores_upper = scores[:,:,:11]
    scores_upper = scores_upper.reshape(-1, 1100)
    scores_all = scores_all.reshape(-1, 1700)

    # get std video confidence score
    video_scores_all = np.mean(scores_all, axis = 1) # (951,)
    video_scores_upper = np.mean(scores_upper, axis = 1) # (951,)
    # standardization
    video_scores_all = MinMaxScaler().fit_transform(video_scores_all.reshape(-1, 1)).ravel()
    video_scores_upper = MinMaxScaler().fit_transform(video_scores_upper.reshape(-1, 1)).ravel()

    return video_scores_all, video_scores_upper


def get_prediction(model, testloader):
    '''
    Get predicted probabilities
    '''
    # prediction
    pred_list, y_true_list = [], []
    with torch.no_grad():
        model.eval()
        for step, (batch_x, batch_y, batch_weight) in enumerate(testloader):
            batch_x = batch_x.to(DEVICE)
            batch_pred = torch.softmax(model(batch_x), dim = 1).cpu().numpy()
            pred_list.append(batch_pred)
            y_true_list.append(batch_y)
    y_pred = np.concatenate(pred_list, axis = 0)
    y_true = np.concatenate(y_true_list, axis = 0)
    return y_pred, y_true


def get_margin(pred, method = "entropy"):
    '''
    Compute margin based on the method assigned
    :pred: prediction of probabilities of all test data after softmax inshape (test_size, n_classes)
    :method: 
        :mean: taking the mean value of the absolute difference from maximal class probability with other probabilities
        :single: compute the margin from the absolute difference of highest and seconed highest class probailites
        :entropy: compute entropy of all class probabilites as the margin
    :return: margin vector
    '''
    margin = np.empty(pred.shape[0])
    n = pred.shape[1]
    if method == "mean":
        for i in range(pred.shape[0]):
            temp_max_prob = np.max(pred[i, :])
            temp_max_prob_index = np.argmax(pred[i, :])
            diff_list = []
            for j in range(n):
                if j == temp_max_prob_index:
                    continue
                diff_list.append(np.abs(temp_max_prob - pred[i, j]))
            margin[i] = np.mean(diff_list)
    elif method == "single":
        for i in range(pred.shape[0]):
            temp_max_prob = np.max(pred[i, :])
            temp_second_highest_prob = np.partition(pred[i, :].flatten(), -2)[-2]
            temp_min_prob = np.min(pred[i, :])
            margin[i] = np.abs(temp_max_prob - temp_second_highest_prob)
    elif method == "entropy":
        for i in range(pred.shape[0]):
            log_prob = np.log2(pred[i, :])
            entropy_vec = np.multiply(log_prob, pred[i,:])
            margin[i] = -np.sum(entropy_vec)
    else:
        raise KeyError("Method is not available!")
    return margin


def get_corr(margin, confidence):
    '''
    Compute correlation coefficient between margin and the attention
    :pred: prediction of probabilities of all test data after softmax in shape (test_size,)
    :confidence: confidence score based on attention in shape (test_size,)
    :return: corr
    '''
    assert margin.shape == confidence.shape
    corr = np.corrcoef(confidence, margin)
    print(f'Corr: {corr[0,1]:10.6f}')
    return corr

def get_all_corr():
    '''
    Compute and all the correlation coefficients with respect to different transformers
    :return: data frame of correlation coefficients
    '''
    import pandas as pd
    corr_list = []
    result_df = pd.DataFrame(data = None, columns = ["base", "large", "huge"], index = [1, 2, 3, 4, 5])
    for name in result_df.columns:
        for i in range(1, 6):
            MODEL_WEIGHT_PATH = '../model_weights/Transformer_' + name + '_' + str(i) + '.pth'
            # load pretrained model
            if name == 'base':
                model = transformer_base()
            elif name == 'large':
                model = transformer_large()
            elif name == 'huge':
                model = transformer_huge()
            model.to(DEVICE)
            weight_path = MODEL_WEIGHT_PATH
            model.load_state_dict(torch.load(weight_path, map_location = DEVICE))

            video_scores_all, video_scores_upper = init_confidence_score()
            _, testloader, test_idx = prepare_data_w_weight()
            pred, _ = get_prediction(model, testloader) # (190, 5)
            # margin = get_margin(pred) # (190,)
            margin = get_margin(pred) # (190,) # updated
            corr = get_corr(margin, video_scores_all[test_idx]) # matrix
            result_df.loc[i, name] = corr[0, 1]

    return result_df

def get_high_quali_pred(model, quantile = [0, 0.2, 0.4, 0.6, 0.8, 1.0]):
    '''
    Get test accuracy with subsets of high quality videos, quality measured by margin
    :model: pretrained transformer default: transformer huge
    :testloader: test dataloader
    :margin: margin value for each test video
    :return: list of test accuracy on those high quality videos
    '''
    from sklearn.metrics import accuracy_score
    import pandas as pd
    _, testloader, _ = prepare_data_w_weight()
    y_pred, y_true = get_prediction(model, testloader)
    margin = get_margin(y_pred, method = "entropy")
    # import matplotlib
    # from matplotlib import pyplot as plt
    # fig, ax = plt.subplots(1, 1, dpi = 270)
    # ax.hist(margin, bins = 200)
    # ax.set_title("Histogram of Entropy")
    quantile_values = np.quantile(margin, quantile)
    # get masks
    mask = np.empty((len(margin), len(quantile)))
    for i in range(len(quantile)):
        mask[:, i] = (margin <= quantile_values[i]) # lower entropy is better!
    # get test accuracy
    y_pred = np.argmax(y_pred, axis = 1) # to dense form
    accuracy_arr = np.empty(len(quantile))
    for i in range(len(quantile)):
        # print(f'Number of high-quality videos: {np.sum(mask[:,i] == True)}')
        accuracy_arr[i] = accuracy_score(y_pred = y_pred[mask[:,i] == True], y_true = y_true[mask[:,i] == True])
    return accuracy_arr



if __name__ == "__main__":
     pass
