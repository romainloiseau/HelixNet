import math
import os.path as osp

import numpy as np
import torch
import torch_scatter
from torch_geometric.data import Data

from .basedataset import BaseDataModule, BaseSequence, BaseSplit
from .laserscan import HelixNetThetaCorrector, LaserScanHNet, SemLaserScanHNet
from .utils.cylindric import cart2cyl, cyl2cart


class HelixNetSequence(BaseSequence):
    _LaserScan_ = LaserScanHNet
    _SemLaserScan_ = SemLaserScanHNet

    def __init__(self, *args, **kwargs):
        super(HelixNetSequence, self).__init__(*args, **kwargs)

        self.theta_corrector = HelixNetThetaCorrector()

        max_delta_rad = (self.theta_corrector._MAX_DIFF_)/2.
        self.maxi_delta_theta = int(
            (self.voxel_res[1] / self.thin_voxel_res[1]) * (math.ceil(max_delta_rad/self.voxel_res[1])))
        self.voxelind_max[1] += 2*self.maxi_delta_theta

    def set_id(self):
        self.id = self.directory.split("/")[-1]

    def __getframe__(self, idx):
        scan, _ = self.__getscan__(idx)

        point_y, point_inst = self.__getlabels__(scan)

        pos = scan.points
        time = scan.time
        frameid = np.zeros(pos.shape[0], dtype=int)

        pos_scan = scan.rtz

        if point_inst is not None:
            return Data(
                pos_scan=pos_scan,
                pos=pos,
                frameid=frameid,
                remissions=scan.remissions,
                point_y=point_y,
                point_inst=point_inst,
                seqid=int(self.id),
                scanid=int(idx),
                time=time,
                fiber=scan.fiber
            )
        return Data(
            pos_scan=pos_scan,
            pos=pos,
            frameid=frameid,
            remissions=scan.remissions,
            point_y=point_y,
            seqid=int(self.id),
            scanid=int(idx),
            time=time,
            fiber=scan.fiber
        )

    def add_fake_frame(self, frame):
        return frame

    def concatenate_frames(self, frame, i, framei):

        frame["pos"] = np.vstack([frame["pos"], framei["pos"]])
        frame["pos_scan"] = np.vstack([frame["pos_scan"], framei["pos_scan"]])

        for field in ["remissions", "time", "fiber"]:
            frame[field] = np.hstack([frame[field], framei[field]])

        frame.frameid = np.hstack([frame.frameid, i+framei.frameid])

        return frame

    def do_feature_aug_(self, pos, time, aug):
        # Rotates the point cloud
        pos[:, 1] += aug[0]

        # Flips the point cloud in the x and/or y axis
        if aug[1][0] == 1:
            pos[:, 1] = pos[:, 1] - math.pi
        elif aug[1][0] == 2:
            pos[:, 1] *= -1
            time *= -1
        elif aug[1][0] == 3:
            pos[:, 1] = math.pi - pos[:, 1]
            time *= -1

        pos = cart2cyl(cyl2cart(pos) * aug[2] + aug[3])

        return pos, time

    def do_feature_aug(self, frame):

        frame.pos_scan, frame.time = self.do_feature_aug_(
            frame.pos_scan, frame.time, self.generate_aug())

        return frame

    def __getitem__(self, idx):

        frame = self.__getallframes__(idx)

        if self.options.feature_aug:
            frame = self.do_feature_aug(frame)

        frame.features, frame.voxelind = self.compute_features(
            frame, frame.frameid)

        frame.pos = torch.from_numpy(frame.pos)

        del frame.pos_scan, frame.remissions, frame.fiber

        frame.backprop = torch.from_numpy(frame.backprop)
        frame.frameid = torch.from_numpy(frame.frameid)
        frame.time = torch.from_numpy(frame.time)
        frame.point_y = frame.point_y.long()

        return frame

    def compute_features(self, frame, frameid, keep=None):

        if self.mode == "train":
            sliceid = (np.floor(self.options.slices_per_rotation * (frame.pos_scan[:, 1] + self.theta_corrector._THETA_CORRECTION_[
                       (frame.fiber).astype(int)]) / (2*math.pi)) % self.options.slices_per_rotation).astype(int)
        else:
            sliceid = np.minimum(
                (self.options.slices_per_rotation * (frame.pos_scan[:, 1] + self.theta_corrector._THETA_CORRECTION_[
                 (frame.fiber).astype(int)]) / (2*math.pi)).astype(int),
                self.options.slices_per_rotation - 1
            )

        polar = torch.from_numpy(
            frame.pos_scan[keep] if keep is not None else frame.pos_scan)
        remissions = torch.from_numpy(
            frame.remissions[keep] if keep is not None else frame.remissions)

        sliceid = torch.from_numpy(
            sliceid[keep] if keep is not None else sliceid)
        sliceframeid = sliceid + self.options.slices_per_rotation * frameid

        slice_theta_ref = torch_scatter.scatter_min(
            torch.from_numpy(frame.time),
            sliceframeid
        )[1]

        slice_theta_ref[slice_theta_ref == polar.shape[0]] = 0
        slice_theta_ref = polar[slice_theta_ref, 1]

        polar_sliceref = polar - torch.cat([
            torch.zeros((polar.size(0), 1)),
            slice_theta_ref[sliceframeid.long()].unsqueeze(-1),
            torch.zeros((polar.size(0), 1))], -1)

        polar_sliceref[:, 2] = torch.maximum(torch.minimum(polar_sliceref[:, 2], torch.tensor(
            self.polar_max_z)), torch.tensor(self.polar_min_z))  # Correct z
        polar_sliceref = polar_sliceref - self.polarsliceref_min_bound  # Correct z
        polar_sliceref[:, 0] = torch.minimum(
            polar_sliceref[:, 0], torch.tensor(self.polar_max_r))  # Correct r

        voxelind = torch.div(polar_sliceref, torch.tensor(
            self.thin_voxel_res), rounding_mode="floor").long()

        gridcenter = (voxelind + 0.5) * self.thin_voxel_res
        polar_sliceref = (polar_sliceref - gridcenter) / \
            (.5*self.thin_voxel_res)

        voxelind[:, 1] = (
            voxelind[:, 1] + self.maxi_delta_theta) % self.voxelind_max[1]

        voxelind = torch.minimum(voxelind, torch.tensor(self.voxelind_max))

        minimax = torch.tensor(
            1.1) if self.options.clipfeat else torch.tensor(1.5)

        polar = torch.stack([
            torch.minimum(polar[:, 0]/self.polar_max_r, minimax),
            torch.cos(polar[:, 1]), torch.sin(polar[:, 1]),
            torch.maximum(torch.minimum(
                torch.cos(polar[:, 1]) * polar[:, 0]/self.polar_max_r, minimax), -minimax),
            torch.maximum(torch.minimum(
                torch.sin(polar[:, 1]) * polar[:, 0]/self.polar_max_r, minimax), -minimax),
            torch.maximum(torch.minimum(2*(polar[:, 2] - self.polar_min_z) /
                                        (self.polar_max_z - self.polar_min_z)-1, minimax), -minimax),
        ], -1)

        features = torch.cat([
            polar_sliceref,
            polar,
            remissions.unsqueeze(-1) / 256.
        ], -1).float()

        voxelind = torch.from_numpy(np.concatenate(
            [np.expand_dims(sliceframeid, -1), voxelind], -1))

        del frame.sliceid

        return features, voxelind


class HelixNetSplit(BaseSplit):
    _DatasetSequence_ = HelixNetSequence

    def create_sequences(self):
        self.sequences = {
            int(seq): self._DatasetSequence_(
                osp.join(
                    self.options.data_dir,
                    "sequences",
                    f"{seq}"
                ),
                self.options,
                self.mode
            ) for seq in self.options.split[self.mode]
        }


class HelixNetDataModule(BaseDataModule):
    _DatasetSplit_ = HelixNetSplit

    def get_features_names(self):
        if self.myhparams.input_dim == 10:
            return [
                "r_centered", "theta_centered", "z_centered",
                "r", "theta_cos", "theta_sin", "x", "y",
                "z", "remission"
            ]
        raise ValueError
