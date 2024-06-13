import copy
from json import detect_encoding
import os
import tempfile
from os import path as osp
from tkinter.messagebox import NO
from unittest.mock import NonCallableMagicMock
import cv2

from tkinter.messagebox import NO
import cv2

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log

from mmdet3d.core import show_multi_modality_result, show_result
from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)
from mmdet3d.datasets.builder import DATASETS
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.pipelines import Compose
from tools.analysis_tools import plot_gt_det_cmp

from mmdet3d.datasets.kitti_dataset import KittiDataset
from tensorboardX import SummaryWriter
import pickle
import numpy as np
import math
from PIL import Image, UnidentifiedImageError, ImageFile

@DATASETS.register_module()
class PlusVADKittiDataset(KittiDataset):
    """
    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list, optional): The range of point cloud used to
            filter invalid predicted boxes.
            Default: [0, -40, -3, 70.4, 40, 0.0].
    """
    CLASSES = ['Pedestrian', 'Cyclist', 'Car', 'Truck']

    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 pts_prefix='pointcloud',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 load_interval=1,
                 used_cameras=4,
                 using_front_center_camera=False,
                 pcd_limit_range=[0, -10, -3, 100.0, 10, 6.0],
                 multi_adj_frame_id=(1,2,1),
                 multi_frame=False,
                 full_data_root=None,
                 camera_names=None,
                 with_vel=False,
                 sequences_split_num=1,
                 keep_consistent_seq_aug=True,
                 rot_range=[-0.3925, 0.3925],
                 scale_ratio_range=[1.0, 1.0],
                 flip_ratio_bev_horizontal=1.0,
                 data_aug_conf=None,
                 scenes_classes=[],
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            pcd_limit_range=pcd_limit_range,
            **kwargs)
        # data_infos is read in the custom_3d.py->Custom3DDataset __init__ function
        # through load_annotations()
        self.multi_adj_frame_id = multi_adj_frame_id
        self.multi_frame = multi_frame
        if self.multi_frame:
            self.full_data_root = full_data_root
        # to load a subset, just set the load_interval in the dataset config
        self.data_infos = self.data_infos[::load_interval]
        
        # data_need_remove = []
        # for item in self.data_infos:
        #     sample_idx = item['image']['image_idx']
        #     image_dict = item.get('image', {})
        #     outer_break = False
        #     for value in image_dict.values():
        #         if isinstance(value, dict):
        #             for img_path in value.values():
        #                 if isinstance(img_path, str):
        #                     img_file = os.path.join(data_root, split, img_path)
        #                     if not os.path.exists(img_file):
        #                         if item not in data_need_remove:
        #                             data_need_remove.append(item)
        #                         outer_break = True
        #                         break
        #                     try:
        #                         with open(img_file, 'rb') as f:
        #                             parser = ImageFile.Parser()
        #                             parser.feed(f.read())
        #                             if not parser.image:
        #                                 if item not in data_need_remove:
        #                                     print(f"Error checking file: {img_file}")
        #                                     data_need_remove.append(item)
        #                                     outer_break = True
        #                                     break
        #                     except Exception as e:
        #                         if item not in data_need_remove:
        #                             print(f"Error checking file: {e}, {img_file}")
        #                             data_need_remove.append(item)
        #                         outer_break = True
        #                         break
        #         if outer_break:
        #             break
            
        # for item in data_need_remove:
        #     if item in self.data_infos:
        #         self.data_infos.remove(item)
                   
        if hasattr(self, 'flag'):
            self.flag = self.flag[::load_interval]
        if camera_names is not None:
            self.camera_names = camera_names
        else:
            self.camera_names = ['front_left_camera', 'front_right_camera',
                                 'side_left_camera', 'side_right_camera',
                                 'rear_left_camera', 'rear_right_camera']
            if using_front_center_camera:
                self.camera_names = ['front_center_camera',
                                     'side_left_camera', 'side_right_camera',
                                     'rear_left_camera', 'rear_right_camera']
            elif used_cameras == 5:
                self.camera_names = ['front_left_camera',
                                     'side_left_camera', 'side_right_camera',
                                     'rear_left_camera', 'rear_right_camera']
            else:
                self.camera_names = ['front_left_camera', 'front_right_camera',
                                     'side_left_camera', 'side_right_camera',
                                    'rear_left_camera', 'rear_right_camera']
                self.camera_names =self.camera_names[:used_cameras]
        self.eval_cnt = 0
        self.with_vel = with_vel
        self.sequences_split_num = sequences_split_num
        self.keep_consistent_seq_aug = keep_consistent_seq_aug
        self.SCENES_CLASSES = scenes_classes

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if not self.test_mode:
                if idx != 0 and self.data_infos[idx]['image']['image_idx'][7:13] != self.data_infos[idx-1]['image']['image_idx'][7:13]:
                    # Not first frame and # of bag_idx is different -> new sequence
                    # print(self.data_infos[idx]['image']['image_idx'][7:13], self.data_infos[idx-1]['image']['image_idx'][7:13])
                    # print(self.data_infos[idx]['image']['image_idx'])
                    curr_sequence += 1
                res.append(curr_sequence)
            else:
                if idx != 0 and self.data_infos[idx]['image']['image_idx'][:33] != self.data_infos[idx-1]['image']['image_idx'][:33]:
                    # Not first frame and # of bag_idx is different -> new sequence
                    # print(self.data_infos[idx]['image']['image_idx'][7:13], self.data_infos[idx-1]['image']['image_idx'][7:13])
                    # print(self.data_infos[idx]['image']['image_idx'])
                    curr_sequence += 1
                res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.sequences_split_num != 1:
            if self.sequences_split_num == "all":
                self.flag = np.array(
                    range(len(self.data_infos)), dtype=np.int64
                )
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(
                            range(
                                0,
                                bin_counts[curr_flag],
                                math.ceil(
                                    bin_counts[curr_flag]
                                    / self.sequences_split_num
                                ),
                            )
                        )
                        + [bin_counts[curr_flag]]
                    )

                    for sub_seq_idx in (
                        curr_sequence_length[1:] - curr_sequence_length[:-1]
                    ):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                # assert (len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.sequences_split_num)
                self.flag = np.array(new_flags, dtype=np.int64)

    def _sample_augmentation(self):
        if self.data_aug_conf is None:
            return None
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                    int(
                        (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
                        * newH
                    )
                    - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                    int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
                    - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def _getitem(self, idx, rot_angle=None, scale_ratio=None, flip=None, flip_direction=None, aug_configs=None):
        if self.test_mode:
            data = self.prepare_test_data(idx)
            return data
        # print(idx)
        data = self.prepare_train_data(idx, rot_angle=rot_angle,
                                    scale_ratio=scale_ratio,
                                    flip=flip,
                                    flip_direction=flip_direction,
                                    aug_configs=aug_configs)
        
        # data = self.prepare_train_data(idx)

        if data is None:
            # print("data is none")
            # print(self.data_infos[idx]['image']['image_idx'])
            idx = (idx + 1) % len(self.data_infos)
            # print(self.data_infos[idx]['image']['image_idx'])
            data = self.__getitem__(idx)
        return data

    def __getitem__(self, idx):
        if isinstance(idx, dict):
            rot_angle, scale_ratio, flip, flip_direction, aug_configs = idx["aug"]
            idx = idx["idx"]
            data = self._getitem(idx, rot_angle, scale_ratio, flip, flip_direction, aug_configs)
        else:
            data = self._getitem(idx)
        # print(data['timestamp'])
        return data


    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file, file_format='pkl')
        if True:
            if not self.test_mode:
                data_infos = list(sorted(data, key=lambda e: (e['point_cloud']['lidar_idx'].split('_')[1], e['point_cloud']['lidar_idx'].split('_')[-1])))
            else:
                data_infos = list(sorted(data, key=lambda e: e['point_cloud']['lidar_idx']))
                # new_data_infos = []
                # for data_info in data_infos:
                #     if int(data_info['point_cloud']['lidar_idx'].split('_')[-1]) % 4 == 0:
                #         new_data_infos.append(data_info)
                # return new_data_infos
            return data_infos #new_data_infos
        if '_' in data[0]['point_cloud']['lidar_idx']:
            data_infos = list(sorted(data, key=lambda e: e['point_cloud']['lidar_idx']))
        else:
            data_infos = list(sorted(data, key=lambda e: int(e['point_cloud']['lidar_idx'].split('.')[0])))

        return data_infos

    def _get_pts_filename(self, idx):
        pts_filename = osp.join(self.root_split, self.pts_prefix, f'{idx}.bin')
        return pts_filename

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        gt_names = set(info['annos']['name'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def get_scenes_ids(self, idx):
        info = self.data_infos[idx]
        scenes_class_label = info['scenes_class_label']
        scenes_ids = []
        for scenes_cat in scenes_class_label:
            if scenes_cat in self.SCENES_CLASSES:
                scenes_ids.append(self.scene_cat2id[scenes_cat])
        return scenes_ids

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
                - difficulty (int): Difficulty defined by KITTI.
                    0, 1, 2 represent xxxxx respectively.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]

        difficulty = info['annos']['difficulty']
        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = self.remove_dontcare(annos)
        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']

        # in imu coordinates
        gt_bboxes_3d = annos['gt_boxes_lidar']
        # gt_bboxes_3d_raw = annos['gt_boxes_lidar']
        # gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d, origin=(0.5, 0.5, 0.5), box_dim=gt_bboxes_3d.shape[1]) # todo

        gt_bboxes = annos['bbox']  # [[1,1,1,1],...,]

        selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
        gt_bboxes = gt_bboxes[selected].astype('float32')
        gt_names = gt_names[selected]

        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        # the index in the CLASSES of every box_label, -1: out of the CLASSES
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        if 'velocity' in annos.keys():
            velocity = annos['velocity']
            assert gt_bboxes_3d.shape[0] == velocity.shape[0]
            gt_bboxes_3d = np.concatenate((gt_bboxes_3d, velocity[:, :2]), axis=1)
            # anns_results.update({'gt_bboxes_velocity': velocity})


        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_names=gt_names,
            plane=None,  # TODO(swc): is it need?
            difficulty=difficulty)
        if 'velocity' in annos.keys():
            anns_results.update({'gt_bboxes_velocity': velocity})
        return anns_results

    def parse_data_info(self, info):
        sample_idx = info['image']['image_idx']
        pts_filename = self._get_pts_filename(sample_idx)
        img_filenames = []
        lidar2img_list = []
        lidar2camera_list = []
        camera_intrinsics_list = []


        for camera_name in self.camera_names:
            # camera_name = camera_name + "_camera"
            img_filename = os.path.join(self.root_split, info['image'][camera_name]['image_path'])
            img_filenames.append(img_filename)

            rect = info['calib'][camera_name]['R0_rect'].astype(np.float64)  # 外参
            Trv2c = info['calib'][camera_name]['Tr_velo_to_cam'].astype(np.float64)  # eye() 没有用到
            P2 = info['calib'][camera_name]['P2'].astype(np.float64)  # 内参
            lidar2img = np.dot(P2, rect)  #

            lidar2img_list.append(lidar2img)
            lidar2camera_list.append(rect)
            camera_intrinsics_list.append(P2)

        lidar2img_list = np.stack(lidar2img_list, axis=0)
        lidar2camera_list = np.stack(lidar2camera_list, axis=0)
        camera_intrinsics_list = np.stack(camera_intrinsics_list, axis=0)
        #todo (syl) add ego2global
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            camera_names=self.camera_names,
            img_info=img_filenames,
            lidar2img=lidar2img_list,
            lidar2camera=lidar2camera_list,
            camera_intrinsics=camera_intrinsics_list)

        if 'ego_pos' in info:
            ego_pos = info['ego_pos']
            input_dict['ego_pos'] = ego_pos

        if 'timestamp' in info:
            timestamp = info['timestamp']
            if isinstance(timestamp, dict):
                timestamp = timestamp['lidar']
            input_dict['timestamp'] = timestamp

        return input_dict

    #todo(syl) parse adj info(camera names, img info, lidar2img, lidar2camera, camera instrinsics)
    def parse_adj_info(self, sample_idx, adj_index):
        if self.test_mode:
            bag_name = sample_idx[:-7] + '.db'
        else:
            bag_name = sample_idx[7:-7] + '.db'
        index_in_bag = int(sample_idx.split('_')[-1])
        adj_index_in_bag = max(index_in_bag - adj_index, 0)

        # ego pose
        # adj_ego_pos_file = os.path.join(self.full_data_root, bag_name, "ego_pos", ("%06d"%adj_index_in_bag)+".npy")
        # ego_pos = np.load(adj_ego_pos_file)
        adj_ego_pos_file = os.path.join(self.full_data_root, bag_name, "ego_pos_with_vel", ("%06d"%adj_index_in_bag)+".pkl")
        with open(adj_ego_pos_file, 'rb') as f:
            ego_pos_info = pickle.load(f, encoding='latin1')
            ego_pos = ego_pos_info['ego_pose']

        # timestamp
        adj_timestamp_file = os.path.join(self.full_data_root, bag_name, "timestamp", ("%06d"%adj_index_in_bag)+".pkl")
        with open(adj_timestamp_file, 'rb') as f:
            timestamp = pickle.load(f)

        # calib
        adj_calib_file = os.path.join(self.full_data_root, bag_name, "calib", ("%06d"%adj_index_in_bag)+".pkl")
        f_adj_calib = open(adj_calib_file, "rb")
        adj_calib = pickle.load(f_adj_calib, encoding='latin1')
        last_line = np.asarray([[0, 0, 0, 1]])

        adj_img_filenames_list = []
        adj_lidar2img_list = []
        adj_lidar2camera_list = []
        adj_camera_intrinsics_list = []

        for index, camera_name in enumerate(self.camera_names):
            adj_img_filename = os.path.join(self.full_data_root, bag_name, camera_name, ("%06d"%adj_index_in_bag)+".png")
            adj_img_filenames_list.append(adj_img_filename)

            rect = np.linalg.inv(
                adj_calib[f'Tr_cam_to_imu_{camera_name}']).astype(np.float64)
            P2 = np.concatenate([adj_calib[f'P_{camera_name}'], last_line], axis=0).astype(np.float64)  # 内参
            lidar2img = np.dot(P2, rect)  #

            adj_lidar2img_list.append(lidar2img)
            adj_lidar2camera_list.append(rect)
            adj_camera_intrinsics_list.append(P2)
        adj_lidar2img_list = np.stack(adj_lidar2img_list, axis=0)
        adj_lidar2camera_list = np.stack(adj_lidar2camera_list, axis=0)
        adj_camera_intrinsics_list = np.stack(adj_camera_intrinsics_list, axis=0)

        adj_info = dict(
            ego_pos=ego_pos,
            timestamp=timestamp,
            img_info=adj_img_filenames_list,
            lidar2img=adj_lidar2img_list,
            lidar2camera=adj_lidar2camera_list,
            camera_intrinsics=adj_camera_intrinsics_list
        )
        return adj_info


    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
        """
        info = self.data_infos[index]
        input_dict = self.parse_data_info(info)
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        if self.multi_frame:
            info_adj_list = self.get_adj_info(input_dict['sample_idx'])
            input_dict.update(dict(adjacent=info_adj_list))
        # print("finish get data info")
        return input_dict

    def get_adj_info(self, sample_idx):
        info_adj_list = []
        for adj_id in range(*self.multi_adj_frame_id):
            #todo(syl) get file_path of history frame
            adj_input_dict = self.parse_adj_info(sample_idx, adj_id)
            info_adj_list.append(adj_input_dict)
        return info_adj_list

    # TODO(swc): only front_left_camera now
    def bbox2result_kitti(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None):
        """Convert 3D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[np.ndarray]): List of array storing the
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str): The prefix of pkl file.
            submission_prefix (str): The prefix of submission file.

        Returns:
            list[dict]: A list of dictionaries with the kitti format.
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to KITTI format')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.data_infos[idx]
            sample_idx = info['image']['image_idx']
            if 'front_left' in info['image']:
                image_shape = info['image']['front_left']['image_shape'][:2]
            else:
                image_shape = info['image']['front_left_camera']['image_shape'][:2]
            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            anno = {
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': [],
                'dt_boxes': [],
                'scores': []
            }
            if len(box_dict['bbox']) > 0:
                box_2d_preds = box_dict['bbox']
                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']

                for box, box_lidar, bbox, score, label in zip(
                        box_preds, box_preds_lidar, box_2d_preds, scores,
                        label_preds):
                    bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(
                        -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                    anno['bbox'].append(bbox)
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)

                    anno['dt_boxes'].append(box_lidar)
                    anno['scores'].append(score)
                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)
            else:
                anno = {
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                    'dt_boxes': np.zeros([0, 7]),
                    'scores': np.array([]),
                }
                annos.append(anno)

            if submission_prefix is not None:
                curr_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(curr_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                                anno['name'][idx], anno['alpha'][idx],
                                bbox[idx][0], bbox[idx][1], bbox[idx][2],
                                bbox[idx][3], dims[idx][1], dims[idx][2],
                                dims[idx][0], loc[idx][0], loc[idx][1],
                                loc[idx][2], anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f)
            # FIXME(swc): sample_idx
            sample_idx = int(sample_idx.split('.')[0])
            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * len(annos[-1]['score']), dtype=np.int64)  # NOTE(swc): [sample_idx] * num_box

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos
    
    
    # TODO(swc): only front_left_camera now
    def convert_valid_bboxes(self, box_dict, info):
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (torch.Tensor): Scores of boxes.
                - labels_3d (torch.Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.

                - bbox (np.ndarray): 2D bounding boxes.
                - box3d_camera (np.ndarray): 3D bounding boxes in
                    camera coordinate.
                - box3d_lidar (np.ndarray): 3D bounding boxes in
                    LiDAR coordinate.
                - scores (np.ndarray): Scores of boxes.
                - label_preds (np.ndarray): Class label predictions.
                - sample_idx (int): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['image']['image_idx']
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)
        if 'front_left' in info['calib']:
            rect = info['calib']['front_left']['R0_rect'].astype(np.float32)
            Trv2c = info['calib']['front_left']['Tr_velo_to_cam'].astype(np.float32)
            P2 = info['calib']['front_left']['P2'].astype(np.float32)
            img_shape = info['image']['front_left']['image_shape']
        else:
            rect = info['calib']['front_left_camera']['R0_rect'].astype(np.float32)
            Trv2c = info['calib']['front_left_camera']['Tr_velo_to_cam'].astype(np.float32)
            P2 = info['calib']['front_left_camera']['P2'].astype(np.float32)
            img_shape = info['image']['front_left_camera']['image_shape']
        P2 = box_preds.tensor.new_tensor(P2)

        box_preds_camera = box_preds.convert_to(Box3DMode.CAM, rect @ Trv2c) # todo bug

        box_corners = box_preds_camera.corners
        box_corners_in_image = points_cam2img(box_corners, P2)
        # box_corners_in_image: [N, 8, 2]
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing
        # check box_preds_camera
        image_shape = box_preds.tensor.new_tensor(img_shape)
        valid_cam_inds = ((box_2d_preds[:, 0] < image_shape[1]) &
                          (box_2d_preds[:, 1] < image_shape[0]) &
                          (box_2d_preds[:, 2] > 0) & (box_2d_preds[:, 3] > 0))
        # check box_preds
        limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
        valid_pcd_inds = ((box_preds.center > limit_range[:3]) &
                          (box_preds.center < limit_range[3:]))
        # valid_inds = valid_cam_inds & valid_pcd_inds.all(-1)
        valid_inds = valid_pcd_inds.all(-1)
        if valid_inds.sum() > 0:
            return dict(
                bbox=box_2d_preds[valid_inds, :].numpy(),
                box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
                box3d_lidar=box_preds[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx)
        else:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)
            

    # NOTE(swc): convert anno_box from lidar_coor(in the pkl) to camera_coor, the same with KITTI
    def anno_lidar2cam(self, anno, calib_info):
        cam_anno = {}
        if 'front_left' in calib_info:
            rect = calib_info['front_left']['R0_rect'].astype(np.float32)
            Trv2c = calib_info['front_left']['Tr_velo_to_cam'].astype(np.float32)
        else:
            rect = calib_info['front_left_camera']['R0_rect'].astype(np.float32)
            Trv2c = calib_info['front_left_camera']['Tr_velo_to_cam'].astype(np.float32)
        
        for k,v in anno.items():
            cam_anno[k] = v
        
        gt_bboxes_lidar = cam_anno['gt_boxes_lidar']
        gt_bboxes_cam= LiDARInstance3DBoxes(gt_bboxes_lidar, origin=(0.5, 0.5, 0.5)).convert_to(Box3DMode.CAM,rect @ Trv2c)
        
        cam_anno['dimensions'] = gt_bboxes_cam.dims.numpy()
        cam_anno['location'] = gt_bboxes_cam.center.numpy()
        cam_anno['rotation_y'] = gt_bboxes_cam.yaw.numpy()
        
        return cam_anno
    
    def bbox2result_pcdet(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None):
        # assert len(net_outputs) == len(self.data_infos), \
        #     'invalid list length of network outputs'
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to pcdet format')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            anno = {
                    'dt_boxes': [],
                    'name': [],
                    'scores': [],
            }

            if 'pts_bbox' in pred_dicts:
                pred_dicts = pred_dicts['pts_bbox']

            if len(pred_dicts['boxes_3d']) > 0:
                scores = pred_dicts['scores_3d']
                box_preds_lidar = pred_dicts['boxes_3d']
                label_preds = pred_dicts['labels_3d']
                if 'boxes_velocity' in pred_dicts:
                    anno.update({'velocity': []})
                    velocity_preds = pred_dicts['boxes_velocity']
                    for box_lidar, score, label, velocity in zip(
                            box_preds_lidar, scores, label_preds, velocity_preds):
                        anno['dt_boxes'].append(box_lidar)
                        anno['name'].append(class_names[int(label)])
                        anno['scores'].append(score)
                        anno['velocity'].append(velocity)
                else:
                    for box_lidar, score, label in zip(
                            box_preds_lidar, scores, label_preds):
                        anno['dt_boxes'].append(box_lidar)
                        anno['name'].append(class_names[int(label)])
                        anno['scores'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)
            else:
                anno = {
                    'dt_boxes': np.zeros([0, 7]),
                    'name': np.array([]),
                    'scores': np.array([]),
                }

                if 'boxes_velocity' in pred_dicts:
                    anno.update({'velocity': np.array([])})

                annos.append(anno)

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 plot_dt_result=False,
                 eval_result_dir=None,
                 eval_file_tail=None,
                 bag_test_flag=False):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str, optional): The prefix of pkl files, including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str, optional): The prefix of submission data.
                If not specified, the submission data will not be generated.
                Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_dict = None
        from mmdet3d.core.evaluation import get_formatted_results, get_formatted_results_with_velocity
        
        gt_annos_pcdet = []
        if not bag_test_flag:
            for info in self.data_infos:
                gt_boxes = info['annos']['gt_boxes_lidar'] 
                gt_names = info['annos']['name']
                gt_anno = {'gt_boxes': gt_boxes, 'name': gt_names}
                if 'velocity' in info['annos']:
                    gt_velocity = info['annos']['velocity']
                    gt_anno.update({'gt_velocity': gt_velocity})
                    assert gt_anno['gt_boxes'].shape[0] == gt_anno['gt_velocity'].shape[0]
                gt_annos_pcdet.append(gt_anno)
        if ('boxes_3d' in results[0]) | ('pts_bbox' in results[0]):
            dets_pcdet = self.bbox2result_pcdet(results, self.CLASSES, pklfile_prefix)
        else:
            dets_pcdet = results
        if plot_dt_result:
                self.save_eval_results(dets_pcdet, gt_annos_pcdet, out_dir) # todo
        # test bag 
        if bag_test_flag:
            self.save_eval_results(dets_pcdet, gt_annos_pcdet, out_dir) # todo
            return result_dict
            
        # to pcdet format
        self.eval_cnt+=10
        if eval_file_tail:
            eval_cnt = eval_file_tail
        else:
            eval_cnt = self.eval_cnt
        result_str, result_dict = get_formatted_results(self.pcd_limit_range, self.CLASSES, gt_annos_pcdet, dets_pcdet, eval_result_dir, eval_cnt, with_vel=self.with_vel)

        print_log('\n' + '****************pcdet eval start.*****************', logger=logger)
        print_log('\n' + result_str, logger=logger)
        print_log('\n' + '****************pcdet eval done.*****************', logger=logger)

        eval_file_name = f'human_readable_results_{eval_cnt}.txt'
        
        if eval_result_dir is not None:
            with open(os.path.join(eval_result_dir, eval_file_name), 'w') as f:
                f.write(result_str)
        return result_dict
    
    @staticmethod
    def concate_img(img0, img1):
        h0,w0=img0.shape[0],img0.shape[1]  #cv2 读取出来的是h,w,c
        h1,w1=img1.shape[0],img1.shape[1]
        h=max(h0,h1)
        w=max(w0,w1)
        org_image=np.ones((h,w,3),dtype=np.uint8)*255
        trans_image=np.ones((h,w,3),dtype=np.uint8)*255

        org_image[:h0,:w0,:]=img0[:,:,:]
        trans_image[:h1,:w1,:]=img1[:,:,:]
        all_image = np.hstack((org_image[:,:w0,:], trans_image[:,:w1,:]))
        return all_image
    
    @staticmethod
    def concat_2dimgs(imgs): #raw_imgs format: width * height * c: 960*540
        
        cam_nums = len(imgs)
        front_image_size = imgs[0].shape[0:2]# H X W:540x960
        side_image_size = imgs[2].shape[0:2] # H X W:540x960
        
        if cam_nums >4:
            rear_image_size = imgs[4].shape[0:2]
        else:
            rear_image_size = (0,0)
        
        new_size = (front_image_size[0]+side_image_size[0]+rear_image_size[0],  # 1080x1920
                    front_image_size[1]+side_image_size[1])
        
        new_img = np.zeros((new_size[0], new_size[1], 3), np.uint8) # HxW
        
        new_img[0:front_image_size[0], 0:front_image_size[1]] = imgs[0]
        new_img[0:front_image_size[0], front_image_size[1]:new_size[1]] = imgs[1]
        
        new_img[front_image_size[0]:new_size[0]-rear_image_size[0], 0:front_image_size[1]] = imgs[2]
        new_img[front_image_size[0]:new_size[0]-rear_image_size[0], front_image_size[1]:new_size[1]] = imgs[3]
        
        if cam_nums >4: # rear img
            offset0 = (front_image_size[1] - rear_image_size[1]) // 2
            offset1 = front_image_size[1] + offset0
            new_img[new_size[0]-rear_image_size[0]:new_size[0], offset0:rear_image_size[1]+offset0] = imgs[4]
            new_img[new_size[0]-rear_image_size[0]:new_size[0], offset1:rear_image_size[1]+offset1] = imgs[5]
        
        new_img = mmcv.imrescale(new_img, 0.6)   
        # mmcv.imshow(new_img)     
        return new_img    
    
    def save_eval_results(self, dets_pcdet, gt_annos_pcdet, results_dir):
        for i, result in enumerate(dets_pcdet):
            data_info = self.data_infos[i]
            pts_path = data_info['point_cloud']['lidar_idx']
            file_name = f"{self.root_split}/{self.pts_prefix}/{pts_path}.bin"
            points = np.fromfile(file_name).reshape(-1, 4)
            if 'front_left' in data_info['image']:
                img_path = data_info['image']['front_left']['image_path']
                camera_names = ['front_left', 'front_right',
                                'side_left', 'side_right',
                                'rear_left', 'rear_right']
            else:
                img_path = data_info['image']['front_left_camera']['image_path']
                camera_names = ['front_left_camera', 'front_right_camera',
                'side_left_camera', 'side_right_camera',
                'rear_left_camera', 'rear_right_camera']
            # camera_names = ['front_left_camera', 'front_right_camera',
            #         'side_left_camera', 'side_right_camera']
            imgs = []
            for cam in camera_names:
                img_path = data_info['image'][cam]['image_path']
                detect_img = f"{self.root_split}/{img_path}"
                detect_img = cv2.imread(detect_img)
                imgs.append(detect_img)
            cat_imgs = self.concat_2dimgs(imgs)
            pred_bboxes = result['dt_boxes']
            gt_boxes = gt_annos_pcdet[i]['gt_boxes']
            scores = result['scores']
            names = result['name']
            path=results_dir + f"/{pts_path}.jpg"
            bev_img = plot_gt_det_cmp(points, gt_boxes, pred_bboxes, self.pcd_limit_range, path=None, scores=scores, names=names)
            all_image = self.concate_img(cat_imgs, bev_img)
            all_image = mmcv.imrescale(all_image, 0.6)   

            cv2.imwrite(path, all_image)
    
    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['point_cloud']['lidar_idx']
            file_name = f"{pts_path}.bin"
            points, img_metas, img = self._extract_data(
                i, pipeline, ['points', 'img_metas', 'img'])
            points = points.numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)

            # multi-modality visualization
            if self.modality['use_camera'] and 'lidar2img' in img_metas.keys():
                img = img.numpy()
                # need to transpose channel to first dim
                img = img.transpose(1, 2, 0)
                show_pred_bboxes = LiDARInstance3DBoxes(
                    pred_bboxes, origin=(0.5, 0.5, 0))
                show_gt_bboxes = LiDARInstance3DBoxes(
                    gt_bboxes, origin=(0.5, 0.5, 0))
                show_multi_modality_result(
                    img,
                    show_gt_bboxes,
                    show_pred_bboxes,
                    img_metas['lidar2img'],
                    out_dir,
                    file_name,
                    box_mode='lidar',
                    show=show)
