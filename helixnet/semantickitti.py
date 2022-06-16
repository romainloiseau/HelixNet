import copy
import math
import os.path as osp

import numpy as np
import torch

from torch_geometric.data import Data

from .basedataset import BaseDataModule, BaseSequence, BaseSplit
from .utils.cylindric import cart2cyl

class SemanticKITTISequence(BaseSequence):

    def __init__(self, *args, **kwargs):
        super(SemanticKITTISequence, self).__init__(*args, **kwargs)

        self.parse_calibration()
        self.parse_poses()
        self.parse_times()
        self.check_data()

    def set_id(self):
        self.id = self.directory.split("/")[-1]

    def check_data(self):
        assert len(self.poses) == len(self.times) == len(
            self.scan_names), "Should have the same number of frames as poses and times"

    def parse_times(self):
        if osp.exists(osp.join(self.directory, "times.txt")):
            self.times = np.loadtxt(osp.join(self.directory, "times.txt"))
        else:
            self.times = 0.1*np.arange(len(self.poses))
        self.dtimes = self.times[1:] - self.times[:-1]
        self.dtimes = np.append(self.dtimes, self.dtimes[-1])
        self.fps = np.mean(self.dtimes)

    """https://github.com/MehmetAygun/4D-PLS/blob/master/datasets/SemanticKitti.py"""
    def parse_calibration(self):
        """ read calibration file with given filename
            Returns
            -------
            dict
                Calibration matrices as 4x4 numpy arrays.
        """
        calib = {}

        calib_file = open(osp.join(self.directory, "calib.txt"))
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()

        self.calib = calib

    def parse_poses(self):
        """ read poses file with per-scan poses from given filename
            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(osp.join(self.directory, "poses.txt"))

        poses = []

        Tr = self.calib["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(
                pose, Tr)).astype(np.float32))

        poses = np.expand_dims(np.stack(poses), 1)
        file.close()

        for i in range(len(poses)):
            if np.abs(poses[i]).max() > 10000:
                j = 1
                while np.abs(poses[i + j]).max() > 10000:
                    j += 1
                poses[i] = copy.deepcopy(poses[i+j])

        self.poses = poses


    def __getmeta__(self, idx):
        meta = {
            "scan_name": osp.join(self.directory, "velodyne", self.scan_names[idx]),
            "has_labels": self.has_labels,
        }

        if self.has_labels:
            meta["label_name"] = osp.join(
                self.directory, "labels", self.label_names[idx])

        return meta

    def __getscan__(self, idx):
        meta = self.__getmeta__(idx)

        if meta["has_labels"]:
            color_dict = self.options.color_map
            nclasses = len(color_dict)
            scan = self._SemLaserScan_(
                nclasses, color_dict
            )
            scan.open_scan(meta["scan_name"])
            scan.open_label(meta["label_name"])
        else:
            scan = self._LaserScan_()
            scan.open_scan(meta["scan_name"])

        return scan, self.poses[idx] if hasattr(self, "poses") else None

    def __getframe__(self, idx):
        scan, pose = self.__getscan__(idx)

        point_y, point_inst = self.__getlabels__(scan)

        pos_scan = scan.points
        remissions = scan.remissions

        frameid = np.zeros(pos_scan.shape[0], dtype=int)
        pos = self.from_pos_scan_to_pos(pos_scan, pose[0])

        if point_inst is not None:
            return Data(
                pos=pos,
                pos_scan=pos_scan,
                frameid=frameid,
                remissions=remissions,
                point_y=point_y,
                point_inst=point_inst,
                pose=pose,
                seqid=int(self.id),
                scanid=int(idx),
                time=np.array([self.times[idx]])
            )
        return Data(
            pos=pos,
            pos_scan=pos_scan,
            frameid=frameid,
            remissions=remissions,
            point_y=point_y,
            pose=pose,
            seqid=int(self.id),
            scanid=int(idx),
            time=np.array([self.times[idx]])
        )

    def from_pos_scan_to_pos(self, pos_scan, pose):
        return np.hstack((pos_scan, np.ones((pos_scan.shape[0], 1), dtype=pos_scan.dtype))).dot(pose[:-1].T)

    def do_feature_aug(self, frame):
        frame.pos_scan = self.do_feature_aug_(frame.pos_scan, self.generate_aug())
        return frame

    def do_feature_aug_(self, pos, aug):
        # Rotates the point cloud
        c, s = np.cos(aug[0]), np.sin(aug[0])
        mat = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=pos.dtype)

        pos = np.dot(pos, mat)
        
        # Flips the point cloud in the x and/or y axis
        if aug[1] == 1:
            pos[:, 1] *= -1
        elif aug[1] == 2:
            pos[:, 0] *= -1
        elif aug[1] == 3:
            pos[:, :1] *= -1

        # Scales the xyz axis of the point cloud
        pos *= aug[2]

        # Shifts the center of the point cloud
        pos += aug[3]

        return pos

    def generate_aug(self):
        if self.options.full_feature_aug:
            return (2*math.pi*np.random.random(), np.random.choice(4, 1), np.random.uniform(0.95, 1.05, (1, 3)), np.random.normal(0, 0.2, (1, 3)))
        return (self.voxel_res[1]*(2*np.random.random()-1), np.random.choice(4, 1), np.random.uniform(0.95, 1.05, (1, 3)), np.random.normal(0, 0.2, (1, 3)))

    def get_debug_idx(self, idx):
        if self.DEBUG:
            idx = self.DEBUGselect[idx]
        return idx

    def __getitem__(self, idx):
        idx = self.get_debug_idx(idx)

        frame = self.__getallframes__(idx)

        frame.pos = self.from_pos_scan_to_pos(frame.pos, np.linalg.inv(frame.pose[0]))
        
        if self.options.feature_aug:            
            frame = self.do_feature_aug(frame)

        frame.time = torch.from_numpy(frame.time).unsqueeze(0)

        frame.pos_scan[:, 2] = np.clip(
            frame.pos_scan[:, 2], self.polar_min_z, self.polar_max_z)
        frame.pos[:, 2] = np.clip(
            frame.pos[:, 2], self.polar_min_z, self.polar_max_z)
        
        frame.features, frame.voxelind = self.compute_features(frame, frame.frameid)

        frame.pos = torch.from_numpy(frame.pos)
        frame.pose = torch.from_numpy(frame.pose)

        del frame.pos_scan, frame.remissions#, frame.time_features#, frame.point_pred

        frame.backprop = torch.from_numpy(frame.backprop)
        frame.frameid = torch.from_numpy(frame.frameid)
        frame.point_y = frame.point_y.long()
        
        return frame

    def add_fake_frame(self, frame):
        frame.time = np.concatenate(
            [frame.time, np.expand_dims(frame.time[0], 0)], -1)
        return frame

    def concatenate_frames(self, frame, i, framei):

        frame["pos"] = np.vstack([frame["pos"], framei["pos"]])
        frame["pos_scan"] = np.vstack([frame["pos_scan"], framei["pos_scan"]])
        frame["remissions"] = np.hstack(
            [frame["remissions"], framei["remissions"]])

        frame.frameid = np.hstack([frame.frameid, i+framei.frameid])
        frame.time = np.concatenate([frame.time, framei.time])

        return frame

    def compute_features(self, frame, frameid, keep=None):

        position = frame.pos_scan[keep] if keep is not None else frame.pos_scan
        remissions = frame.remissions[keep] if keep is not None else frame.remissions

        polar = cart2cyl(position)

        sliceid = np.minimum((self.options.slices_per_rotation*(.5 + (
            polar[:, 1] / (2*math.pi)))).astype(int), self.options.slices_per_rotation-1)
        sliceframeid = sliceid + self.options.slices_per_rotation * frameid

        polar_sliceref = polar - self.slice_theta_ref[sliceid]
        polar_sliceref = polar_sliceref - self.polarsliceref_min_bound
        polar_sliceref[:, 0] = np.minimum(polar_sliceref[:, 0], self.polar_max_r)

        voxelind = (polar_sliceref / self.thin_voxel_res).astype(int)
        voxelind = np.minimum(voxelind, self.voxelind_max)

        gridcenter = (voxelind + 0.5) * self.thin_voxel_res

        polar_sliceref = (polar_sliceref - gridcenter) / (.5*self.thin_voxel_res)

        minimax = torch.tensor(1.1) if self.options.clipfeat else torch.tensor(1.5)
        polar = torch.from_numpy(polar)

        features = torch.cat([
            torch.from_numpy(polar_sliceref),
            torch.minimum(polar[:, 0].unsqueeze(-1)/self.polar_max_r, minimax),
            torch.cos(polar[:, 1]).unsqueeze(-1), torch.sin(polar[:, 1]).unsqueeze(-1),
            2*(polar[:, 2].unsqueeze(-1) - self.polar_min_z) /
            (self.polar_max_z - self.polar_min_z)-1,
            torch.maximum(torch.minimum(torch.from_numpy(position[:, :2]/self.polar_max_r), minimax), -minimax),
            torch.from_numpy(remissions).unsqueeze(-1)
        ], -1).float()

        voxelind = torch.from_numpy(np.concatenate(
            [np.expand_dims(sliceframeid, -1), voxelind], -1))

        return features, voxelind

class SemanticKITTISplit(BaseSplit):
    _DatasetSequence_ = SemanticKITTISequence

    def create_sequences(self):
        self.sequences = {
            int(seq): self._DatasetSequence_(
                osp.join(
                    self.options.data_dir,
                    "sequences",
                    "%02d" % seq
                ),
                self.options,
                self.mode
            ) for seq in self.options.split[self.mode]
        }
            
    def get_sequence(self, idx):
        return self.sequences[idx]

class SemanticKITTIDataModule(BaseDataModule):
    _DatasetSplit_ = SemanticKITTISplit

    def get_features_names(self):
        if self.myhparams.input_dim == 10:
            return [
                "r_centered", "theta_centered", "z_centered",
                "r", "theta_cos", "theta_sin",
                "z", "x", "y",
                "remission"
            ]
        raise ValueError
