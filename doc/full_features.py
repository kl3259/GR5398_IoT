import numpy as np

from src.settings import EVALUATION_VIDEO_PATH


def extract_trajectories(keypoints, with_index=False):
    """
    with_index: For True, return the index of frames that have features
    """
    trajectory = []
    index = []
    for i, (_, k) in enumerate(keypoints):
        if len(k) != 0:
            index.append(i)
            two_d_point = k[0, [0, 1], :]
            trajectory.append(two_d_point)

    if with_index:
        return np.stack(trajectory), index
    else:
        return np.stack(trajectory)


def traj_interp(traj, n_frames=100):
    n_samples = traj.shape[0]

    if n_samples == 0:
        raise ValueError("trajectories of length 0!!")

    result = np.empty((n_frames, 2, 17))

    traj = np.asarray(traj)
    dest_x = np.linspace(0, 100, n_frames)
    src_x = np.linspace(0, 100, n_samples)

    for i in range(2):
        for j in range(17):
            result[:, i, j] = np.interp(
                dest_x,
                src_x,
                traj[:, i, j]
            )

    return result.reshape(-1)


def get_full_feature_data(feature_dirs, transform=None, **kwargs):
    features = []
    labels = []
    label_encoder = {'no_interaction': 0, 'open_close_fridge': 1,
                     'put_back_item': 2, 'screen_interaction': 3, 'take_out_item': 4}

    for path in feature_dirs:
        label = path.split('/')[-2]
        d = np.load(path, allow_pickle=True)
        traj = extract_trajectories(d['keypoints'], with_index=False)
        if transform:
            traj = transform(traj, **kwargs)
        labels.append(label_encoder.get(label, None))
        features.append(traj)

    return features, np.stack(labels)


def get_correctly_ordered_prediction_filelist():
    import os
    PATH = EVALUATION_VIDEO_PATH

    filelist = []
    for root, dirs, files in os.walk(os.path.abspath(PATH)):
        for file in files:
            filelist.append(os.path.join(root, file))

    filelistunordered = []
    for i in filelist:
        if '.mp4' in i:
            filelistunordered.append(i)

    filelist_evalids = []
    for i in filelistunordered:
        splitfilelist = i.split("_")
        filelist_evalids.append(int(splitfilelist[len(splitfilelist)-2]))

    filelist_evalids.sort()

    predictionfiles = []
    for i in filelist_evalids:
        for j in filelistunordered:
            if j.find(str(i)) > 0:
                predictionfiles.append(j)
    return predictionfiles