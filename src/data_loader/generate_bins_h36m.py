import argparse
import os
import torch.utils.data as data
import numpy as np
import pickle
import sys

train_subjects = [1, 5, 6, 7, 8]
test_subjects = [9, 11]
# train_path = '../../data/annot_h36m_cam_reg/annot_cam_train.pickle'
# test_path = '../../data/annot_h36m_cam_reg/annot_cam_test.pickle'


def compute_mean_std(pts_3d):
    mean_3d = np.mean(pts_3d, axis=0)
    std_3d = np.std(pts_3d, axis=0)

    mean_3d = mean_3d.reshape(1, -1)
    std_3d = std_3d.reshape(1, -1)

    return mean_3d, std_3d


def compute_extremities(points_3d_norm):

    tol_frac = 1.1  # should be greater than 1

    if points_3d_norm.shape[1] >= 48:
        points_3d_x = points_3d_norm[:, ::3]
        points_3d_y = points_3d_norm[:, 1::3]
        points_3d_z = points_3d_norm[:, 2::3]
    else:
        points_3d_x = points_3d_norm[:, :, 0].reshape(-1)
        points_3d_y = points_3d_norm[:, :, 1].reshape(-1)
        points_3d_z = points_3d_norm[:, :, 2].reshape(-1)

    min_x = np.min(points_3d_x) * tol_frac
    min_y = np.min(points_3d_y) * tol_frac
    min_z = np.min(points_3d_z) * tol_frac
    max_x = np.max(points_3d_x) * tol_frac
    max_y = np.max(points_3d_y) * tol_frac
    max_z = np.max(points_3d_z) * tol_frac

    return min_x, min_y, min_z, max_x, max_y, max_z


def normalise_pose(pts_3d, mean_3d, std_3d):
    eps = 1e-08

    pts_3d_norm = np.divide(pts_3d - mean_3d, eps + std_3d)

    return pts_3d_norm


def get_train_stats(pts_3d_norm, opt):
    n_joints = opt.n_joints

    min_x, min_y, min_z, max_x, max_y, max_z = compute_extremities(pts_3d_norm)

    train_stats = dict()
    train_stats['min_x'] = min_x
    train_stats['min_y'] = min_y
    train_stats['min_z'] = min_z

    train_stats['max_x'] = max_x
    train_stats['max_y'] = max_y
    train_stats['max_z'] = max_z

    return train_stats


def generate_bins(split, opt, scale=1000., pose_stats={}):

    print('Processing {} split'.format(split))

    n_joints = opt.n_joints

    annot_file_path = os.path.join(opt.data_dir, 'annot_cam_{}.pickle'.format(split))
    if split == 'train':
        with open(annot_file_path, 'rb') as f:
            annot = pickle.load(f)
        allowed_subj_list = train_subjects
    else:
        with open(annot_file_path, 'rb') as f:
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

    n_data = annot['id'].shape[0]

    # points 3d normalization
    pts_3d = np.copy(annot['joint_3d_rel'])
    pts_3d = pts_3d.reshape(pts_3d.shape[0], 16 * 3)

    if 'mean_3d' not in pose_stats.keys():
        print('Computing mean and std of poses')
        mean_3d, std_3d = compute_mean_std(pts_3d.copy())
    else:
        print('using supplied mean and std of poses')
        mean_3d = pose_stats['mean_3d']
        std_3d = pose_stats['std_3d']

    if opt.use_sc_norm is False:
        pts_3d_norm = normalise_pose(pts_3d.copy(), mean_3d, std_3d)
    else:
        pts_3d_norm = pts_3d / scale

    if 'tr_st' not in pose_stats.keys():
        print('computing min max stats')
        tr_st = get_train_stats(pts_3d_norm.copy(), opt)
    else:
        print('using supplied min max stats')
        tr_st = pose_stats['tr_st']

    min_x = tr_st['min_x']
    min_y = tr_st['min_y']
    min_z = tr_st['min_z']
    max_x = tr_st['max_x']
    max_y = tr_st['max_y']
    max_z = tr_st['max_z']

    pts_3d_x = pts_3d_norm[:, ::3]
    pts_3d_y = pts_3d_norm[:, 1::3]
    pts_3d_z = pts_3d_norm[:, 2::3]

    index_x = (((pts_3d_x - min_x) * opt.n_bin_x) / (max_x - min_x)).astype(int)
    index_x[index_x == opt.n_bin_x] = opt.n_bin_x - 1
    index_x[index_x < 0] = 0
    assert np.sum(index_x < opt.n_bin_x) == n_data * n_joints and np.sum(index_x >= 0) == n_data * n_joints

    index_y = (((pts_3d_y - min_y) * opt.n_bin_y) / (max_y - min_y)).astype(int)
    index_y[index_y == opt.n_bin_y] = opt.n_bin_y - 1
    index_y[index_y < 0] = 0
    assert np.sum(index_y < opt.n_bin_y) == n_data * n_joints and np.sum(index_y >= 0) == n_data * n_joints

    index_z = (((pts_3d_z - min_z) * opt.n_bin_z) / (max_z - min_z)).astype(int)
    index_z[index_z == opt.n_bin_z] = opt.n_bin_z - 1
    index_z[index_z < 0] = 0
    assert np.sum(index_z < opt.n_bin_z) == n_data * n_joints and np.sum(index_z >= 0) == n_data * n_joints

    index_x_onehot = np.zeros((index_x.shape[0], index_x.shape[1], opt.n_bin_x), dtype='uint8')
    for i in range(opt.n_bin_x):
        index_x_onehot[:, :, i] = (index_x == i).astype('uint8')

    index_y_onehot = np.zeros((index_y.shape[0], index_y.shape[1], opt.n_bin_y), dtype='uint8')
    for i in range(opt.n_bin_y):
        index_y_onehot[:, :, i] = (index_y == i).astype('uint8')

    index_z_onehot = np.zeros((index_z.shape[0], index_z.shape[1], opt.n_bin_z), dtype='uint8')
    for i in range(opt.n_bin_z):
        index_z_onehot[:, :, i] = (index_z == i).astype('uint8')

    annot_bins = dict()
    annot_bins['bin_x'] = index_x
    annot_bins['bin_y'] = index_y
    annot_bins['bin_z'] = index_z
    annot_bins['bin_x_oh'] = index_x_onehot
    annot_bins['bin_y_oh'] = index_y_onehot
    annot_bins['bin_z_oh'] = index_z_onehot

    print('Saving bin info')

    output_file_name = os.path.join(opt.save_dir, 'annot_bins_sc_norm_{}_{}_{}_{}_{}.pkl'.format(
        opt.use_sc_norm, opt.n_bin_x, opt.n_bin_y, opt.n_bin_z, split))

    with open(output_file_name, 'wb') as f:
        pickle.dump(annot_bins, f)

    pose_stats = dict()
    pose_stats['mean_3d'] = mean_3d
    pose_stats['std_3d'] = std_3d
    pose_stats['tr_st'] = tr_st

    return pose_stats


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/annot_h36m_cam_reg', help='dir to save bin info')
    parser.add_argument('--save_dir', type=str, default='../data/annot_h36m_cam_reg', help='dir to save bin info')
    parser.add_argument('--n_bin_x', type=int, default=10, help='no. bins in X directions')
    parser.add_argument('--n_bin_y', type=int, default=10, help='no. bins in Y directions')
    parser.add_argument('--n_bin_z', type=int, default=10, help='no. bins in Z directions')
    parser.add_argument('--n_joints', type=int, default=16, help='no. joints in pose')
    parser.add_argument('--use_sc_norm', action='store_true', help='use simple scale normalisation')

    opt = parser.parse_args()

    scale = 1000.

    train_pose_stats = generate_bins('train', opt, scale,)
    generate_bins('test', opt, scale, pose_stats=train_pose_stats)




