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
    pred_list = []
    with torch.no_grad():
        model.eval()
        for step, (batch_x, batch_y, batch_weight) in enumerate(testloader):
            batch_x = batch_x.to(DEVICE)
            batch_pred = torch.softmax(model(batch_x), dim = 1).cpu().numpy()
            pred_list.append(batch_pred)
    pred = np.concatenate(pred_list, axis = 0)
    return pred


def get_margin(pred):
    '''
    Compute margin by taking the mean value of the absolute difference from maximal class probability with other probabilities
    :pred: prediction of probabilities of all test data after softmax inshape (test_size, n_classes)
    :return: margin vector
    '''
    margin = np.empty(pred.shape[0])
    n = pred.shape[1]
    for i in range(pred.shape[0]):
        temp_max_prob = np.max(pred[i, :])
        temp_max_prob_index = np.argmax(pred[i, :])
        diff_list = []
        for j in range(n):
            if j == temp_max_prob_index:
                continue
            diff_list.append(np.abs(temp_max_prob - pred[i, j]))
        margin[i] = np.mean(diff_list)
    return margin

def get_margin_alt(pred):
    '''
    Compute margin by taking the mean value of the absolute difference from maximal class probability with other probabilities
    :pred: prediction of probabilities of all test data after softmax inshape (test_size, n_classes)
    :return: margin vector
    '''
    margin = np.empty(pred.shape[0])
    n = pred.shape[1]
    for i in range(pred.shape[0]):
        temp_max_prob = np.max(pred[i, :])
        temp_min_prob = np.min(pred[i, :])
        margin[i] = np.abs(temp_max_prob - temp_min_prob)
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
            pred = get_prediction(model, testloader) # (190, 5)
            # margin = get_margin(pred) # (190,)
            margin = get_margin_alt(pred) # (190,)
            corr = get_corr(margin, video_scores_all[test_idx]) # matrix
            result_df.loc[i, name] = corr[0, 1]

    return result_df



if __name__ == "__main__":
     pass
