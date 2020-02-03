import torch.utils.data as data
import numpy as np
import pickle
import sys
import cv2
import os

train_subjects = [1,5,6,7,8]
test_subjects = [9,11]
train_path = '../../data/annot_h36m_cam_reg/annot_cam_train.pickle'
test_path = '../../data/annot_h36m_cam_reg/annot_cam_test.pickle'

def compute_stats(pts_3d):
    mean_3d = np.mean(pts_3d, axis=0)
    std_3d = np.std(pts_3d, axis=0)

    return mean_3d, std_3d

def compute_extremeties(points_3d_norm):
    points_3d_x = points_3d_norm[:,::3]
    points_3d_y = points_3d_norm[:, 1::3]
    points_3d_z = points_3d_norm[:, 2::3]
    min_x = np.min(points_3d_x)
    min_y = np.min(points_3d_y)
    min_z = np.min(points_3d_z)
    max_x = np.max(points_3d_x)
    max_y = np.max(points_3d_y)
    max_z = np.max(points_3d_z)
    return min_x,min_y,min_z,max_x,max_y,max_z

def getTrainStats(njoints):
    with open(train_path, 'rb') as f:
        annot_train = pickle.load(f)
    allowed_subj_list = train_subjects
    annot_train['joint_3d_rel'] = annot_train['joint_3d_mono'] - annot_train['joint_3d_mono'][:, 6:7, :]
    subj_mask = np.zeros(annot_train['id'].shape[0], dtype='bool')
    for subj_id in allowed_subj_list:
        subj_mask = np.logical_or(subj_mask, annot_train['subject'] == subj_id)

    valid_mask = subj_mask

    valid_ids = np.arange(annot_train['id'].shape[0])[valid_mask]

    for key in annot_train.keys():
        annot_train[key] = annot_train[key][valid_ids]

    pts_3d = np.copy(annot_train['joint_3d_rel'])
    pts_3d = pts_3d.reshape(pts_3d.shape[0], njoints * 3)
    mean_3d, std_3d = compute_stats(pts_3d)
    eps = 1e-8
    pts_3d_norm = np.divide(pts_3d - mean_3d, eps + std_3d)
    min_x, min_y, min_z, max_x, max_y, max_z = compute_extremeties(pts_3d_norm)
    return mean_3d,std_3d,min_x, min_y, min_z, max_x, max_y, max_z


def generate_bins(split, njoints, num_x, num_y, num_z):
    mean_3d,std_3d,min_x, min_y, min_z, max_x, max_y, max_z = getTrainStats(njoints)
    if split=='train':
        with open(train_path,'rb') as f:
            annot = pickle.load(f)
        allowed_subj_list = train_subjects

    else:
        with open(test_path,'rb') as f:
            annot = pickle.load(f)
        allowed_subj_list = test_subjects

    annot['joint_3d_rel'] = annot['joint_3d_mono'] - annot['joint_3d_mono'][:, 6:7, :]
    subj_mask = np.zeros(annot['id'].shape[0], dtype='bool')
    for subj_id in allowed_subj_list:
        subj_mask = np.logical_or(subj_mask, annot['subject'] == subj_id)

    valid_mask = subj_mask

    valid_ids = np.arange(annot['id'].shape[0])[valid_mask]

    for key in annot.keys():
        annot[key] = annot[key][valid_ids]

    # points 3d normalization
    pts_3d = np.copy(annot['joint_3d_rel'])
    pts_3d = pts_3d.reshape(pts_3d.shape[0], 16 * 3)

    eps = 1e-8
    pts_3d_norm = np.divide(pts_3d - mean_3d, eps + std_3d)
    pts_3d_x = pts_3d_norm[:,::3]
    pts_3d_y = pts_3d_norm[:,1::3]
    pts_3d_z = pts_3d_norm[:,2::3]
    index_x = ((pts_3d_x - min_x)*num_x/(max_x-min_x)).astype(int)
    index_x[index_x>=num_x] = num_x-1
    index_x[index_x<0] = 0
    index_y = ((pts_3d_y- min_y) * num_y / (max_y - min_y)).astype(int)
    index_y[index_y >= num_y] = num_y - 1
    index_y[index_y < 0] = 0
    index_z = ((pts_3d_z - min_z) * num_z / (max_z - min_z)).astype(int)
    index_z[index_z >= num_z] = num_z - 1
    index_z[index_z < 0] = 0
    index_x_onehot = np.zeros((index_x.shape[0], index_x.shape[1],num_x))
    for i in range(num_x):
        index_x_onehot[:,:,i] = (index_x == i).astype(int)
    index_y_onehot = np.zeros((index_y.shape[0], index_y.shape[1], num_y))
    for i in range(num_y):
        index_y_onehot[:, :, i] = (index_y == i).astype(int)
    index_z_onehot = np.zeros((index_z.shape[0], index_z.shape[1], num_z))
    for i in range(num_z):
        index_z_onehot[:, :, i] = (index_z == i).astype(int)

    annot_bins = dict()
    annot_bins['bin_x'] = index_x
    annot_bins['bin_y'] = index_y
    annot_bins['bin_z'] = index_z
    annot_bins['bin_x_oh'] = index_x_onehot
    annot_bins['bin_y_oh'] = index_y_onehot
    annot_bins['bin_z_oh'] = index_z_onehot
    with open('annot_bins_'+split+'_'+str(num_x)+'_'+str(num_y)+'_'+str(num_z)+'.pkl','wb') as f:
        pickle.dump(annot_bins,f)


if __name__=="__main__":
    # num_x = sys.argv[1]
    # num_y = sys.argv[2]
    # num_z = sys.argv[3]
    num_x = 10
    num_y = 10
    num_z = 10
    njoints = 16
    generate_bins('train',njoints,num_x,num_y,num_z)
    generate_bins('test',njoints,num_x,num_y,num_z)






