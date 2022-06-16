import copy
import math
import os
import os.path as osp
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch_scatter
from hydra.utils import to_absolute_path
from matplotlib import cm
from plyfile import PlyData, PlyElement
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader
from tqdm.auto import tqdm

from .laserscan import LaserScan, SemLaserScan
from .utils.projection import do_up_projection
from .utils.semantickitti import apply_learning_map, from_sem_to_color

NDEBUG = 50

class BaseSequence:
    _LaserScan_ = LaserScan
    _SemLaserScan_ = SemLaserScan

    def __init__(self, directory, options, mode):
        self.directory = directory
        self.set_id()

        self.mode = mode
        self.options = options

        self.set_file_names()
        self.init_euclidian2polar()
        

        self.set_DEBUG()

        if self.mode=="test" and self.options.save_test_preds:
            self.prepare_test_dir()

    def init_euclidian2polar(self):
        self.slice_ids = np.arange(self.options.slices_per_rotation)

        self.slice_theta_ref = np.stack([
            np.zeros(self.options.slices_per_rotation),
            2*math.pi * self.slice_ids/self.options.slices_per_rotation - math.pi,
            np.zeros(self.options.slices_per_rotation)
        ]).T
        self.voxel_res = np.array(self.options.voxel_res)
        self.voxel_res[1] = math.pi * self.voxel_res[1] / 180

        assert len(self.options.split_hierarchy) == self.options.n_hierarchy

        self.thin_voxel_res = self.voxel_res / np.prod(np.array(self.options.split_hierarchy), 0)#np.array(self.options.split_hierarchy)**self.options.n_hierarchy

        self.polar_min_z = self.options.polar_min_z
        self.polar_max_z = self.options.polar_max_z
        self.polar_max_r = self.options.polar_max_r
        self.voxelind_max = np.array([self.polar_max_r / self.thin_voxel_res[0], 2*math.pi / (self.thin_voxel_res[1] * self.options.slices_per_rotation) - 1, (self.polar_max_z - self.polar_min_z) / self.thin_voxel_res[2]]).astype(int)
        
        self.polarsliceref_min_bound = np.array(
            [0, 0, self.polar_min_z])

    def prepare_test_dir(self):
        if not os.path.exists("sequences"):
            os.system("mkdir sequences")
            f = open("sequences/description.txt", "w")
            f.write(f"project url: \n")
            f.write(f"publication url: \n")
            f.write(f"bibtex: \n")
            f.close()
        os.system(f"mkdir sequences/{self.id}")
        os.system(f"mkdir sequences/{self.id}/predictions")

    def set_DEBUG(self):
        self.DEBUG = self.options.DEBUG and self.mode not in ["train"]
        if self.DEBUG:
            if self.options.DEBUG != True:
                self.DEBUGselect = np.arange(min(len(self.scan_names), self.options.DEBUG)).astype(np.int64)
            else:
                self.DEBUGselect = np.linspace(
                    0, len(self.scan_names) - 1, NDEBUG).astype(np.int64)

    def set_id(self):
        raise NotImplementedError

    def set_file_names(self):
        self.scan_names = sorted(os.listdir(
            osp.join(self.directory, "velodyne")))

        self.has_labels = osp.exists(
            osp.join(self.directory, "labels")) and self.options.use_semantics
        if self.has_labels:
            self.label_names = sorted(os.listdir(
                osp.join(self.directory, "labels")))
            assert len(self.label_names) == len(
                self.scan_names), "Should have the same number of frames as labels"

    def __len__(self):
        if self.DEBUG:
            return len(self.DEBUGselect)
        return len(self.scan_names)

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
        raise NotImplementedError

    def __getallframes__(self, idx):
        frame = self.__getframe__(idx)

        for frameid, di in enumerate(self.options.temporal_scan_sequence):
            if self.mode == "train" or not self.options.eval_in_online_setup:
                new_idx = idx + di
            else:
                new_idx = -1
                
            if new_idx >= 0 and new_idx < len(self.scan_names):
                frame = self.concatenate_frames(
                    frame, 1+frameid, self.__getframe__(new_idx))
            else:
                frame = self.add_fake_frame(frame)
        frame.backprop = frame.frameid == 0
        return frame

    def __getlabels__(self, scan):
        point_y, point_inst = None, None
        if hasattr(scan, "sem_label"):
            point_y = torch.from_numpy(apply_learning_map(
                scan.sem_label.astype(np.int16),
                self.options.learning_map
            ))
        else:
            point_y = torch.zeros(scan.points.shape[0]).int()
        if self.options.use_instances:
            point_inst = torch.from_numpy(scan.inst_label.astype(np.int32))
        return point_y,point_inst

    def from_pos_scan_to_pos(self, pos_scan, pose):
        return np.hstack((pos_scan, np.ones((pos_scan.shape[0], 1), dtype=pos_scan.dtype))).dot(pose[:-1].T)

    def generate_aug(self):
        if self.options.full_feature_aug:
            return (2*math.pi*np.random.random(), np.random.choice(4, 1), np.random.uniform(0.95, 1.05, (1, 3)), np.random.normal(0, 0.2, (1, 3)))
        return (self.voxel_res[1]*(2*np.random.random()-1), np.random.choice(4, 1), np.random.uniform(0.95, 1.05, (1, 3)), np.random.normal(0, 0.2, (1, 3)))

class BaseSplit(Dataset):
    _DatasetSequence_ = BaseSequence

    def __init__(self, options, mode: str = "train"):
        self.options = copy.deepcopy(options)
        self.options.feature_aug = self.options.feature_aug and mode == "train"
        self.mode = mode

        self.create_sequences()
        self.frames = []
        for seqid, seq in self.sequences.items():
            self.frames += [(seqid, frame) for frame in range(len(seq))]

    def __repr__(self):
        o = f"{self.__class__.__name__} {self.mode} with {len(self.sequences)} sequences\n"
        sequences = [len(seq) for seq in self.sequences.values()]
        sequencessum = np.sum(sequences)
        o += f"Size of sequences: {sequencessum} <== {sequences}"
        return o

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        sequence, scan = self.frames[idx]
        return self.sequences[sequence][scan]

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

class BaseDataModule(pl.LightningDataModule):
    _DatasetSplit_ = BaseSplit

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.myhparams = SimpleNamespace(**kwargs)
        self.myhparams.data_dir = to_absolute_path(self.myhparams.data_dir)

    def __repr__(self):
        o = [f"{self.__class__.__name__}"]
        for dataset in ["train", "val", "test"]:
            if hasattr(self, f"{dataset}_dataset"):
                o.append(getattr(self, f"{dataset}_dataset").__repr__())
        return "\n".join(o)

    def get_features_names(self):
        return [
            "r_centered", "theta_centered", "z_centered",
            "r", "theta_cos", "theta_sin",
            "z", "x", "y",
            "remission"
        ]

    def from_labels_to_color(self, labels):
        return from_sem_to_color(apply_learning_map(labels, self.myhparams.learning_map_inv), self.myhparams.color_map)

    def setup(self, stage: str = None):
        if stage in (None, 'fit'):
            self.train_dataset = self._DatasetSplit_(self.myhparams, "train")
            self.val_dataset = self._DatasetSplit_(self.myhparams, "valid")
        elif stage in (None, 'validate'):
            self.val_dataset = self._DatasetSplit_(self.myhparams, "valid")

        if stage in (None, 'test'):
            self.test_dataset = self._DatasetSplit_(self.myhparams, "test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.myhparams.batch_size,
            shuffle=True,
            num_workers=self.myhparams.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.myhparams.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.myhparams.num_workers
        )

    def show(self, item, color="y", view="2d"):
        item = copy.deepcopy(item)

        if view == "3d":
            self.show_3d(item, color)
        elif view == "2d":
            self.show_2d(item, color)
        else:
            raise ValueError

    def get_color_from_item(self, item, color):
        if color == "y" and hasattr(item, "point_y"):
            color = self.from_labels_to_color(item.point_y).numpy()
        elif color == "pred_y" and hasattr(item, "point_pred_y"):
            color = self.from_labels_to_color(item.point_pred_y).numpy()
        elif color == "inst" and hasattr(item, "point_inst"):
            color = cm.get_cmap("tab20")(
                np.unique(item.point_inst, return_inverse=True)[1])
        elif color == "slice":
            color = (255*cm.get_cmap("tab20c")((4*item.voxelind[:, 0])% 20)).astype(int)
        elif color == "voxel":
            voxelind = torch.div(item.voxelind, torch.tensor([1] + list(np.prod(np.array(self.myhparams.split_hierarchy), 0))), rounding_mode="floor")
            
            maxi = (voxelind.max(0)[0] + 1).numpy()
            voxelind = (voxelind[:, 1:].numpy() * np.array([np.prod(maxi[2:]), np.prod(maxi[3:]), 1])).sum(-1)
            
            voxelind = np.random.permutation(np.prod(maxi[1:]))[voxelind] % 4
            voxelind = (voxelind + 4*item.voxelind[:, 0].numpy()) % 20
            color = (255*cm.get_cmap("tab20c")(voxelind)).astype(int)
        elif color == "theta" and hasattr(item, "rtz"):
            theta = item.rtz[:, 1]
            color = (255*cm.get_cmap("tab20c")((theta + math.pi)/(2*math.pi))).astype(int)
        elif color == "fiber" and hasattr(item, "fiber"):
            color = (255*cm.get_cmap("tab20c")(item.fiber%20)).astype(int)
        elif color == "time" and hasattr(item, "time"):
            time = (item.time - item.time.min()) / (item.time.max() - item.time.min())
            color = (255*cm.get_cmap("jet")(time.numpy())).astype(int)
        else:
            color = (255*cm.get_cmap("viridis")(item.features[:, -1])).astype(int)
        return color

    def show_2d(self, item, color):
        color_names = color.split(";")
        colors = [self.get_color_from_item(item, c) for c in color_names]

        n_seqs = 1 + len(self.myhparams.temporal_scan_sequence)
        plt.figure(figsize=(7*n_seqs, 7*len(color_names)))
        
        if hasattr(item, "voxelind"):
            frameid = torch.div(item.voxelind[:, 0], self.myhparams.slices_per_rotation, rounding_mode='floor')
        else:
            frameid = item.frameid

        pos = item.pos - item.pos[item.backprop].mean(0) * torch.tensor([1, 1, 0])

        for frame in range(frameid.max().int().item() + 1):
            isframe = frameid == frame
            proj = do_up_projection(pos[isframe])

            for i, (cname, thiscolor) in enumerate(zip(color_names, colors)):
                if cname in ["y", "pred_y"] and thiscolor.shape[0] != pos.shape[0]:
                    if frame == 0:
                        proj = do_up_projection(pos[item.backprop])
                        image = thiscolor[proj]
                    else:
                        proj = do_up_projection(pos[isframe])
                        image = 255*np.stack([proj < 0 for _ in range(3)], -1)
                else:
                    proj = do_up_projection(pos[isframe])
                    image = thiscolor[isframe][proj]

                if image is not None:
                    image[proj < 0] = 255
                    plt.subplot(len(color_names), n_seqs, (n_seqs-frame)+i*n_seqs)
                    plt.imshow(image)
                    plt.title(f"{cname} frame {frame}")
                    
                    plt.axis("off")

        plt.show()

    def get_label_from_raw_feature(self, c):
        if c == "y":
            return "Ground truth"
        elif c == "pred_y":
            return "Prediction"
        elif c == "slice":
            return "Slice"
        elif c == "voxel":
            return "Partition"
        elif c == "time":
            return "Acquisition time"
        return c

    def show_3d(self, item, color, voxelize=True, has_error=True):
        has_error = has_error and hasattr(item, "point_pred_y")
        
        if item.pos.size(0) != item.point_y.size(0):
            for attr in ["pos", "sliceid", "frameid", "features", "voxelind", "time"]:
                if hasattr(item, attr) and len(getattr(item, attr)) == len(item.backprop):
                    setattr(item, attr, getattr(item, attr)[item.backprop])

        title = f"Frame {item.scanid} from Sequence {item.seqid}"
        if has_error:
            item.error = torch.logical_and(item.point_y != item.point_pred_y, item.point_y != 0)
            accuracy = (item.point_y[item.point_y != 0] == item.point_pred_y[item.point_y != 0]).float().mean().item()
            title += f"                    Accuracy = {100*accuracy:.2f}%"

        if voxelize:
            choice = torch.unique((item.pos / 0.25).int(), return_inverse=True, dim=0)[1]
            for attr in ["sliceid", "frameid", "voxelind", "time"]:
                if hasattr(item, attr) and len(getattr(item, attr)) == len(choice):
                    setattr(item, attr, torch_scatter.scatter_max(getattr(item, attr), choice, 0)[0])
            for attr in ["point_y", "point_pred_y", "point_inst"]:
                if hasattr(item, attr):
                    setattr(item, attr, torch_scatter.scatter_sum(F.one_hot(getattr(item, attr)), choice, 0).argmax(-1))
            for attr in ["pos", "features"]:
                if hasattr(item, attr):
                    setattr(item, attr, torch_scatter.scatter_mean(getattr(item, attr), choice, 0))
            if has_error:
                item.error = torch.logical_and(item.point_y != item.point_pred_y, item.point_y != 0)

        if item.pos.shape[0] > 2**15:
            choice = np.random.choice(item.pos.shape[0], 2**15, replace=False)
            for attr in ["pos", "point_y", "point_pred_y", "point_inst", "sliceid", "frameid", "features", "error", "voxelind", "time"]:
                if hasattr(item, attr):
                    setattr(item, attr, getattr(item, attr)[choice])
        
        datadtype = torch.float16
        item.pos -= item.pos.mean(0)
        margin = int(0.02 * 600)
        layout = go.Layout(
            title=title,
            width=1200,
            height=600,
            scene=dict(aspectmode='data', ),
            margin=dict(l=margin, r=margin, b=margin, t=4*margin),
            uirevision=True,
            showlegend=False
        )
        fig = go.Figure(
            layout=layout,
            data=go.Scatter3d(
                x=item.pos[:, 0].to(datadtype), y=item.pos[:, 1].to(datadtype), z=item.pos[:, 2].to(datadtype),
                mode='markers',
                marker=dict(size=2, color=self.get_color_from_item(item, color.split(";")[0])))
        )
        if has_error:
            error_pos = item.pos[item.error]
            fig.add_trace(
                go.Scatter3d(
                    opacity=0.,
                    x=error_pos[:, 0].to(datadtype),
                    y=error_pos[:, 1].to(datadtype),
                    z=error_pos[:, 2].to(datadtype),
                    mode='markers',
                    marker=dict(
                        size=2,
                        color="red"),
                    visible=True,
                    showlegend=False,
                )
            )

        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=[{"marker": dict(size=2, color=self.get_color_from_item(item, c))}, [0]],
                        label=self.get_label_from_raw_feature(c),
                        method="restyle"
                    ) for c in color.split(";")
                ]),
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.05,
            xanchor="left",
            y=0.88,
            yanchor="top"
            ),
        ]
        if has_error:
            updatemenus.append(
                dict(
                    buttons=[dict(
                        method='restyle',
                        label='Show error',
                        visible=True,
                        args=[{'opacity': [1., 0.], 'showlegend': False}],
                        args2=[{'opacity': [0.2, 1.], 'showlegend': False}],
                        #args=[{'visible': True, 'showlegend': False}, [1]],
                        #args2=[{'visible': False, 'showlegend': False}, [1]],
                        )],
                    pad={'r': 10, 't': 10},
                    showactive=True,
                    type='buttons',
                    xanchor='left',
                    x=0.05,
                    yanchor='top',
                    y=0.95, ),
            )

        fig.update_layout(updatemenus=updatemenus)
        fig.update_layout(scene_aspectmode='data')

        # Place the legend on the left
        fig.update_layout(
            legend=dict(
                yanchor="middle",
                y=0.5,
                xanchor="right",
                x=0.99))

        # Hide all axes and no background
        fig.update_layout(
            scene=dict(
                xaxis_title='',
                yaxis_title='',
                zaxis_title='',
                xaxis=dict(
                    autorange=True,
                    showgrid=False,
                    ticks='',
                    showticklabels=False,
                    backgroundcolor="rgba(0, 0, 0, 0)"
                ),
                yaxis=dict(
                    autorange=True,
                    showgrid=False,
                    ticks='',
                    showticklabels=False,
                    backgroundcolor="rgba(0, 0, 0, 0)"
                ),
                zaxis=dict(
                    autorange=True,
                    showgrid=False,
                    ticks='',
                    showticklabels=False,
                    backgroundcolor="rgba(0, 0, 0, 0)"
                )
            )
        )

        fig.show()

    def save_as_ply(self, item, filename=None, cfg=None, do_3d_tube=False, remove_unlabeled=True):

        if filename is None:
            filename = item.filename.split("/")[-1][:-5] + "ply"
            if do_3d_tube:
                filename = "tube" + filename

        vertex_val = self.get_vertex_val(item, cfg, do_3d_tube, remove_unlabeled)[0]

        PlyData([PlyElement.describe(vertex_val, 'vertex')]).write(filename)
        print("SAVED !")

    def get_vertex_val(self, item, cfg, do_3d_tube, remove_unlabeled=False):
        dtypes = [
            ('x', 'float32'), ('y', 'float32'), ('z', 'float32'),
            ('label', 'int16'), ('frame', 'int32'), ('batch', 'int32'), ('slice', 'int32'),
            ('islice', 'int32'), ('ir', 'int32'), ('itheta', 'int32'), ('iz', 'int32'),
            ('ztime', 'float32'), ('time', 'float32'),
            ('Red', 'uint8'), ('Green', 'uint8'), ('Blue', 'uint8')
        ]

        
        if do_3d_tube:
            dtypes += [(f'keep', 'bool')]
            anchor_point = item.pos[np.random.choice(item.pos.size(0))]
            if cfg is not None:
                r2 = cfg.model.transformer.restrict_attention_distance**2
            else:
                r2 = 10**2
            tube = ((item.pos - anchor_point)**2).sum(-1) < r2
        else:
            tube = item.frameid


        scalez = 0.075 * 20

        ztime = -(item.pos[:, 2] - (item.voxelind[:, 0]) / scalez)
        usebatch = (hasattr(item, "batch") and item.batch is not None)
        slice = item.voxelind[:, 0] + (item.batch if usebatch else 0) * (item.voxelind[:, 0].max() + 1)

        maxi, mini = item.pos.min(0)[0], item.pos.max(0)[0]


        size = item.point_y.size(0)
        concat = torch.cat([
            item.pos[:size],
            item.point_y.unsqueeze(-1),
            item.frameid.unsqueeze(-1)[:size], item.batch.unsqueeze(-1)[:size] if usebatch else item.frameid.unsqueeze(-1)[:size], slice.unsqueeze(-1)[:size],
            item.voxelind[:size],
            ztime.unsqueeze(-1)[:size], item.time.unsqueeze(-1)[:size],
            (256*((item.pos-mini)/(maxi-mini))).to(torch.uint8)[:size]
        ], -1)

        if hasattr(item, "point_pred_y"):
            concat = torch.cat([concat, item.point_pred_y.unsqueeze(-1)], -1)
            dtypes.append(('prediction', 'int16'))

        

        if do_3d_tube:
            concat = torch.cat([concat, tube.unsqueeze(-1)], -1)[tube]

        elif remove_unlabeled:
            concat = concat[item.point_y != 0]

        out = concat.numpy()
        
        
        out = np.array([tuple(o) for o in tqdm(out, leave=False)],
            dtype=dtypes
            )

        
        return out, dtypes
