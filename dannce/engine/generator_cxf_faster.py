import os
import numpy as np
from tensorflow import keras
from dannce.engine import processing_cxf as processing
from dannce.engine import ops as ops
from dannce.engine.video import LoadVideoFrame
import warnings
import time
import scipy.ndimage.interpolation
import tensorflow as tf
from typing import Union
from lilab.cameras_setup import get_view_xywh_wrapper
from dannce.engine.com_detection_cxf import ims_to_com2ds, matlab_pose_to_cv2_pose, com2ds_to_com3d
import torch

# from tensorflow_graphics.geometry.transformation.axis_angle import rotate
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
from typing import List, Dict, Tuple, Text
import ffmpegcv
from lilab.mmdet_dev.canvas_reader import CanvasReader
from lilab.cvutils_new.canvas_reader_pannel import CanvasReaderPannelMask, CanvasReaderPannel
from dannce.engine.generator_cxf import (DataGenerator_3Dconv_torch_video_canvas_faster, 
                                         DataGenerator_3Dconv_torch,
                                         sample_grid_ravel,
                                         rectify_com3d_nan)
from multiprocessing import Process, Queue

conn_ronly, conn_wonly = multiprocessing.Pipe()


def producer_vid(videopath, pkldata, gpu):
    vid = CanvasReaderPannel(videopath, pkldata, gpu=gpu)
    # ret, ims_pannel = vid.read()
    # im_pannels_nclass = np.array(ims_pannel)
    # buffer = im_pannels_nclass.tobytes()
    while True:
        ret, ims_pannel = vid.read()
        if ret:
            im_pannels_nclass = np.array(ims_pannel)
            buffer = im_pannels_nclass.tobytes()
            conn_wonly.send_bytes(buffer)
            # q.put(im_pannels_nclass)
        else:
            # q.put(None)
            break
        

class DataGenerator_3Dconv_torch_video_canvas_ultrafaster_single(DataGenerator_3Dconv_torch_video_canvas_faster):
    def set_video(self, videopath, gpu, pkldata):
        self.pro = Process(target=producer_vid, args=(videopath, pkldata, gpu))
        self.pro.start()

        assert os.path.exists(videopath), "Video file not found"
        assert self.batch_size == 1, "Batch size must be 1 for video data"

        self.vid = CanvasReaderPannel(videopath, pkldata, gpu=gpu)
        self.nframes = len(self.vid)
        ret, ims_pannel = self.vid.read()
        im_pannels_nclass = np.array(ims_pannel)
        self.vid_buffershape = im_pannels_nclass.shape
        self.vid_bufferlen = len(im_pannels_nclass.tobytes())
        self.vid.release()
        self.iframe =0

        self.nclass = 1
        self.batch_size = self.nclass
        self.n_channels_in = 1 #gray
        self.gpu = gpu
        self.device = torch.device(f"cuda:{gpu}")

        coms_3d = self.vid.pkl_data['coms_3d']
        coms_3d = coms_3d.reshape(-1, coms_3d.shape[-1])
        self.init_grids(coms_3d)

    def release(self):
        self.pro.terminate()
        conn_ronly.close()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        indexes = np.arange(self.batch_size) + index * self.batch_size
        list_frames_temp = [f'0_{i}' for i in indexes]
        X, y = self.__data_generation(list_frames_temp)
        return X, y

    def __data_generation(self, list_IDs_temp):
        self.batch_size = self.nclass
        # assert conn_ronly.poll(0.1)
        buffer = conn_ronly.recv_bytes(self.vid_bufferlen)
        im_pannels_nclass = np.frombuffer(buffer, dtype=np.uint8).reshape(self.vid_buffershape)

        assert im_pannels_nclass is not None
        assert len(im_pannels_nclass) == self.batch_size == len(list_IDs_temp) == 1
        assert not self.depth
        assert self.mode == "3dprob"
        assert self.mono

        X, xgrid_roi, y_3d = [], [], []
        for i, ID in enumerate(list_IDs_temp):
            experimentID = int(ID.split("_")[0])
            assert experimentID == 0
            ims = im_pannels_nclass[i]
            if 'coms_3d' in self.vid.pkl_data:
                com_3d = self.vid.pkl_data['coms_3d'][self.iframe,i,:]
                self.iframe += 1
            else:
                coms_2d = ims_to_com2ds(ims)
                com_3d = com2ds_to_com3d(coms_2d, self.ba_pose)
            [X_each, xgrid_roi_each], y_3d_each = self.quary_gridsample_by_com3d(com_3d, ims)
            X.append(X_each)
            xgrid_roi.append(xgrid_roi_each)
            y_3d.append(y_3d_each)
        X = np.stack(X, axis=0)
        return [X, xgrid_roi], y_3d
    

def producer_vid_mulitanimals(videopath, pkldata, gpu):
    vid = CanvasReaderPannelMask(videopath, pkldata, gpu=gpu)
    while True:
        im_pannels_nclass = vid.read_canvas_mask_img_out()
        if len(im_pannels_nclass):
            im_pannels_nclass = np.array(im_pannels_nclass)
            buffer = im_pannels_nclass.tobytes()
            conn_wonly.send_bytes(buffer)
            # q.put(im_pannels_nclass)
        else:
            # q.put(None)
            break

class DataGenerator_3Dconv_torch_video_canvas_ultrafaster(DataGenerator_3Dconv_torch):
    def set_video(self, videopath, gpu, pkldata):
        assert os.path.exists(videopath), "Video file not found"
        assert self.batch_size == 1, "Batch size must be 1 for video data"
        
        self.pro = Process(target=producer_vid_mulitanimals, args=(videopath, pkldata, gpu))
        self.pro.start()

        self.vid = CanvasReaderPannelMask(videopath, pkldata, gpu=gpu)
        self.nframes = len(self.vid)
        im_pannels_nclass = self.vid.read_canvas_mask_img_out()
        im_pannels_nclass = np.array(im_pannels_nclass)
        self.vid_buffershape = im_pannels_nclass.shape
        self.vid_bufferlen = len(im_pannels_nclass.tobytes())


        self.nclass = 2
        self.batch_size = self.nclass
        self.n_channels_in = 1 #gray
        self.gpu = gpu
        self.device = torch.device(f"cuda:{gpu}")

        coms_3d = self.vid.pkl_data['coms_3d']
        coms_3d = rectify_com3d_nan(coms_3d)
        coms_3d = coms_3d.reshape(-1, coms_3d.shape[-1])
        self.init_grids(coms_3d)

    def release(self):
        self.pro.terminate()
        conn_ronly.close()

    def __init__(self, *args, **kwargs):
        self.iframe = 0
        super().__init__(*args, **kwargs)

    def __len__(self):
        return self.nframes

    def __getitem__(self, index):
        indexes = np.arange(self.batch_size) + index * self.batch_size
        list_frames_temp = [f'0_{i}' for i in indexes]
        X, y = self.__data_generation(list_frames_temp)
        return X, y

    def __data_generation(self, list_IDs_temp):
        self.batch_size = self.nclass
        buffer = conn_ronly.recv_bytes(self.vid_bufferlen)
        im_pannels_nclass = np.frombuffer(buffer, dtype=np.uint8).reshape(self.vid_buffershape)
        assert len(im_pannels_nclass) == self.batch_size == len(list_IDs_temp)
        assert not self.depth
        assert self.mode == "3dprob"
        assert self.mono

        X, xgrid_roi, y_3d = [], [], []
        for i, ID in enumerate(list_IDs_temp):
            experimentID = int(ID.split("_")[0])
            assert experimentID == 0
            ims = im_pannels_nclass[i]
            if 'coms_3d' in self.vid.pkl_data:
                com_3d = self.vid.pkl_data['coms_3d'][self.iframe,i,:]
            else:
                coms_2d = ims_to_com2ds(ims)
                com_3d = com2ds_to_com3d(coms_2d, self.ba_pose)

            [X_each, xgrid_roi_each], y_3d_each = self.quary_gridsample_by_com3d(com_3d, ims) #cxf
            X.append(X_each)
            xgrid_roi.append(xgrid_roi_each)
            y_3d.append(y_3d_each)
        self.iframe += 1
        X = np.stack(X, axis=0)
        return [X, xgrid_roi], y_3d

    def init_grids(self, com_3ds):
        vstep = (self.vmax - self.vmin) / self.nvox
        assert com_3ds.shape[1]==3
        Xaxis_min, Yaxis_min, Zaxis_min = com_3ds.min(axis=0)
        Xaxis_max, Yaxis_max, Zaxis_max = com_3ds.max(axis=0)
        xgrid = np.arange(Xaxis_min+1.2*self.vmin, Xaxis_max+1.2*self.vmax, vstep)
        ygrid = np.arange(Yaxis_min+1.2*self.vmin, Yaxis_max+1.2*self.vmax, vstep)
        zgrid = np.arange(Zaxis_min+1.2*self.vmin, Zaxis_max+1.2*self.vmax, vstep)
        (x_coord_3d, y_coord_3d, z_coord_3d) = np.meshgrid(xgrid, ygrid, zgrid)
        grid_flatten_3d = np.stack((x_coord_3d.ravel(), y_coord_3d.ravel(), z_coord_3d.ravel()),axis=1)
        experimentID=0

        camParams = [self.camera_params[experimentID][name] for name in self.camnames[experimentID]]
        proj_grid_voxel_ncam = []
        for camParam in camParams:
            proj_grid = ops.project_to2d(grid_flatten_3d, camParam["K"], camParam["R"], camParam["t"])
            proj_grid = proj_grid[:, :2]
            if self.distort:
                proj_grid = ops.distortPoints(
                    proj_grid,
                    camParam["K"],
                    np.squeeze(camParam["RDistort"]),
                    np.squeeze(camParam["TDistort"]),
                ).T
            proj_grid_voxel = np.reshape(proj_grid, [*x_coord_3d.shape, 2]).astype('float16')
            proj_grid_voxel_ncam.append(proj_grid_voxel)
            
        self.grid_1d = (xgrid, ygrid, zgrid)
        self.grid_coord_3d = np.stack([x_coord_3d, y_coord_3d, z_coord_3d], axis=-1).astype('float16')  #(nvox_y, nvox_x, nvox_z, 3)
        self.proj_grid_voxel_ncam = np.array(proj_grid_voxel_ncam).astype('int16')  #(ncam, nvox_y, nvox_x, nvox_z, 2)
        self.ncam = len(self.proj_grid_voxel_ncam)
        self.checked_imsize = False
        # self.threadpool = ThreadPool(self.ncam)
        # context = multiprocessing.get_context(method='fork')
        # self.processpool = context.Pool(self.ncam)

    def quary_gridsample_by_com3d(self, com_3d, ims):
        # input=gray, output=gray.
        assert len(com_3d)==3
        assert len(ims) == self.ncam
        com_index = np.array([np.searchsorted(self.grid_1d[i], com_3d[i], side='right')
                        for i in range(3)])  #(3,)
        com_range = np.floor(com_index[:,None] + [- self.nvox/2, self.nvox/2]).astype(int) #(3,2)

        xgrid_roi = self.grid_coord_3d[None,
                                        com_range[1][0]:com_range[1][1], 
                                        com_range[0][0]:com_range[0][1], 
                                        com_range[2][0]:com_range[2][1],
                                        :]   #(1, nvox_y, nvox_x, nvox_z, 3)

        if not self.checked_imsize:
            self.proj_grid_voxel_ncam_indravel = np.zeros([*self.proj_grid_voxel_ncam.shape[:-1], 1], dtype='int64')
            for i in range(self.ncam):
                im = ims[i]
                assert len(im.shape)==3 and im.shape[2]==1
                assert im.data.c_contiguous
                np.clip(self.proj_grid_voxel_ncam[i, ..., 0], 0, im.shape[1] - 1, out=self.proj_grid_voxel_ncam[i, ..., 0])
                np.clip(self.proj_grid_voxel_ncam[i, ..., 1], 0, im.shape[0] - 1, out=self.proj_grid_voxel_ncam[i, ..., 1])

                indravel = np.ravel_multi_index(np.moveaxis(self.proj_grid_voxel_ncam[i, ..., ::-1], -1, 0), im.shape[:2])
                self.proj_grid_voxel_ncam_indravel[i] = indravel[..., None]

            self.checked_imsize = True

        proj_grid_voxel_ncam_indravel_roi = self.proj_grid_voxel_ncam_indravel[:,
                                        com_range[1][0]:com_range[1][1], 
                                        com_range[0][0]:com_range[0][1], 
                                        com_range[2][0]:com_range[2][1],
                                        :]   #(ncam, nvox_y, nvox_x, nvox_z, 1)
        arglist = [(ims[icam], proj_grid_voxel_ncam_indravel_roi[icam], 'nearest')
                        for icam in range(self.ncam)]
        result = [sample_grid_ravel(*args) for args in arglist]

        X = np.empty(
            (1, self.ncam, self.nvox, self.nvox, self.nvox),
            dtype="float32",
        )
        for icam in range(self.ncam):
            r, g, b = result[icam]
            X[0, icam] = r
        # X = np.stack([result[icam][0] for icam in range(self.ncam)], axis=-1)[None,...].astype(np.float32)

        # assert self.norm_im
        X = processing.preprocess_3d(X)
        
        y_3d = np.empty((1, self.nvox, self.nvox, self.nvox, 14), dtype="float32")
        X = np.moveaxis(X, 1, -1) #(1, self.nvox, self.nvox, self.nvox, self.ncam)

        # X = np.empty((1, self.nvox, self.nvox, self.nvox, self.ncam), dtype="float32")
        return [X[0], xgrid_roi[0]], y_3d[0]