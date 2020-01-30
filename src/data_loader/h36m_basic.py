import torch
import torch.utils.data as data
import numpy as np
import pickle
import cv2
import os
import math

train_subjects = [1, 5, 6, 7, 8]
test_subjects = [9, 11]


class H36M(data.Dataset):
    def __init__(self, opt, split, train_stats, allowed_subj_list=None):
        print('==> initializing H36M {} data.'.format(split))
        print('data dir {}'.format(opt.data_dir))
        annot_file = os.path.join(opt.data_dir, 'annot_cam_' + ('train' if split == 'train' else 'test') + '.pickle')

        print ('==> loading annot file {}.'.format(annot_file))
        annot = {}

        print(annot_file)
        with open(annot_file, 'rb') as f:
            annot = pickle.load(f)

        # creating root relative
        annot['joint_3d_rel'] = annot['joint_3d_mono'] - annot['joint_3d_mono'][:, 6:7, :]

        self.opt = opt
        self.annot = annot
        self.split = split
        self.allowed_subject_list = allowed_subj_list
        self.train_stats = train_stats
        self.n_joints = self.annot['joint_3d_rel'].shape[1]

        self.img_mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)  # img net mean
        self.img_std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)  # img net var

        subj_mask = np.zeros(annot['id'].shape[0], dtype='bool')

        if allowed_subj_list is None:
            if split == 'train':
                allowed_subj_list = train_subjects
            else:
                allowed_subj_list = test_subjects

        for subj_id in allowed_subj_list:
            subj_mask = np.logical_or(subj_mask, annot['subject'] == subj_id)

        valid_mask = subj_mask

        valid_ids = np.arange(self.annot['id'].shape[0])[valid_mask]

        for key in annot.keys():
            self.annot[key] = self.annot[key][valid_ids]

        # points 3d normalization
        pts_3d = np.copy(annot['joint_3d_rel'])
        pts_3d = pts_3d.reshape(pts_3d.shape[0], 16 * 3)

        if 'mean_3d' not in self.train_stats.keys():
            print('Computing Mean and Std')
            self.mean_3d, self.std_3d = self.compute_mean_std()
            self.train_stats['mean_3d'] = self.mean_3d
            self.train_stats['std_3d'] = self.std_3d
        else:
            print('Using pre-computed Mean and Std')
            self.mean_3d = self.train_stats['mean_3d']
            self.std_3d = self.train_stats['std_3d']

        eps = 1e-8
        pts_3d_norm = np.divide(pts_3d - self.mean_3d, eps + self.std_3d)
        self.annot['joint_3d_normalized'] = pts_3d_norm

        self.n_samples = self.annot['id'].shape[0]

        print('Loaded {} with {} labelled samples'.format(split, self.n_samples))

        self.shuffle_ids = np.arange(0, self.n_samples)

    def compute_mean_std(self):
        pts_3d = np.copy(self.annot['joint_3d_rel'])
        pts_3d = pts_3d.reshape(pts_3d.shape[0], self.n_joints * 3)

        mean_3d = np.mean(pts_3d, axis=0)
        std_3d = np.std(pts_3d, axis=0)

        return mean_3d, std_3d

    def load_image(self, index):
        subj = self.annot['subject'][index]
        act = self.annot['action'][index]
        sub_act = self.annot['subaction'][index]
        cam_id = self.annot['camera'][index]
        id = self.annot['id'][index]

        folder = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}'.format(subj, act, sub_act, cam_id)
        path = '{}/{}/{}_{:06d}.jpg'.format(self.opt.img_dir, folder, folder, id)
        img = cv2.imread(path)

        if img is None:
            print('** ' + path + ' **')

        if img.shape[0] != self.opt.inp_img_size or img.shape[1] != self.opt.inp_img_size:
            img = cv2.resize(img, (self.opt.inp_img_size, self.opt.inp_img_size))

        img = img.astype(np.float32)

        img = img / 256.

        return img

    def get_inputs(self, index):
        id = index
        img = self.load_image(id)

        if img.ndim < 3:
            img = np.reshape(img, (img.shape[0], img.shape[1], 1))

        if self.opt.res_norm is True:
            img = (img - self.img_mean) / self.img_std

        pose = np.copy(self.annot['joint_3d_normalized'][id])
        pose_un = np.copy(self.annot['joint_3d_rel'][id])

        subj = self.annot['subject'][id]

        return id, img, pose, pose_un, subj

    def __getitem__(self, index):

        shuff_id = self.shuffle_ids[index]

        id, img, pose, pose_un, subj = self.get_inputs(shuff_id)

        meta = dict()
        meta['annot_id'] = id
        meta['subj'] = subj

        return img, pose, pose_un, meta

    def __len__(self):
        return self.n_samples

    def init_epoch(self, split):
        print('Shuffling {} data'.format(split))
        np.random.shuffle(self.shuffle_ids)
