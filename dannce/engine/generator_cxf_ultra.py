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
from ffmpegcv.video_info import get_info
from lilab.cvutils_new.canvas_reader_pannel import CanvasReaderPannelMask
from dannce.engine.generator_cxf import (DataGenerator_3Dconv_torch,
                                         sample_grid_ravel,
                                         rectify_com3d_nan)
from dannce.my_extension_voxel import voxel_sampling, voxel_sampling_c_last
from multiprocessing import Queue


NFRAME = 10

class CanvasReaderPannelNoblock:
    def __init__(self, videopath, pkldata, gpu, out_numpy_shape):
        self.out_numpy_shape = out_numpy_shape
        self.shared_array = multiprocessing.Array('b', int(NFRAME*np.prod(self.out_numpy_shape)))
        self.np_array = np.frombuffer(self.shared_array.get_obj(), dtype=np.uint8).reshape((NFRAME,*self.out_numpy_shape))
        self.q = Queue(maxsize=(NFRAME-1)*2)
        process = multiprocessing.Process(target=child_process, 
                                        args=(self.shared_array, self.q, videopath, pkldata, gpu, out_numpy_shape))
        process.start()
        self.process = process
        self.iframe = 0

    def release(self):
        self.process.terminate()

    def read(self):
        self.q.get()
        data_id = self.q.get()
        if data_id is None:
            return False, None
        else:
            self.iframe += 1
            return True, self.np_array[data_id]
        

def child_process(shared_array, q:Queue, videopath:str, pkldata, gpu, out_numpy_shape):
    vid = CanvasReaderPannelMask(videopath, pkldata, gpu=gpu)
    np_array = np.frombuffer(shared_array.get_obj(), dtype=np.uint8).reshape((NFRAME,*out_numpy_shape))
    for i in range(len(vid)):
        img = vid.read_canvas_mask_img_out()
        iloop = i % NFRAME
        assert img.shape == tuple(out_numpy_shape)
        q.put(True)
        np_array[iloop] = img
        q.put(iloop)
    q.put(True)
    q.put(None)

        
class DataGenerator_3Dconv_torch_video_canvas_ultrafaster(DataGenerator_3Dconv_torch):
    def set_video(self, videopath, gpu, pkldata):
        assert os.path.exists(videopath), "Video file not found"
        assert self.batch_size == 1, "Batch size must be 1 for video data"

        self.vid = CanvasReaderPannelMask(videopath, pkldata, gpu=gpu)
        self.nframes = len(self.vid)
        out_numpy_shape=self.vid.out_numpy_shape
        self.vid.release()

        self.vidinfo = get_info(videopath)
        self.nframes = self.vidinfo.count
        self.vidnoblock = CanvasReaderPannelNoblock(videopath, pkldata, gpu=gpu, out_numpy_shape=out_numpy_shape)
        self.iframe =0

        self.nclass = out_numpy_shape[0]

        self.batch_size = self.nclass
        self.n_channels_in = 1 #gray
        self.gpu = gpu
        self.device = torch.device(f"cuda:{gpu}")

        self.pkldata = pkldata
        coms_3d = pkldata['coms_3d']
        coms_3d = rectify_com3d_nan(coms_3d)
        coms_3d = coms_3d.reshape(-1, coms_3d.shape[-1])
        self.init_grids(coms_3d)


    def release(self):
        self.vidnoblock.release()

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
        ret, im_pannels_nclass = self.vidnoblock.read()
        assert len(im_pannels_nclass) == self.batch_size == len(list_IDs_temp)
        assert not self.depth
        assert self.mode == "3dprob"
        assert self.mono

        X, xgrid_roi, y_3d = [], [], []
        for i, ID in enumerate(list_IDs_temp):
            experimentID = int(ID.split("_")[0])
            assert experimentID == 0
            ims = im_pannels_nclass[i]
            if 'coms_3d' in self.pkldata:
                com_3d = self.pkldata['coms_3d'][self.iframe,i,:]
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


    def quary_gridsample_by_com3d(self, com_3d, ims):
        # input=gray, output=gray.
        assert len(com_3d)==3
        assert len(ims) == self.ncam
        # return ([np.zeros((self.nvox, self.nvox, self.nvox,9), dtype=np.float32), 
        #         np.zeros((self.nvox, self.nvox, self.nvox, 3), dtype=np.float32)], 
        #         np.empty((self.nvox, self.nvox, self.nvox, 14), dtype=np.float32))
        com_index = np.array([np.searchsorted(self.grid_1d[i], com_3d[i], side='right')
                        for i in range(3)])  #(3,)
        com_range = np.floor(com_index[:,None] + [- self.nvox/2, self.nvox/2]).astype(int) #(3,2)

        xgrid_roi = self.grid_coord_3d[ com_range[1][0]:com_range[1][1], 
                                        com_range[0][0]:com_range[0][1], 
                                        com_range[2][0]:com_range[2][1],
                                        :]   #(nvox_y, nvox_x, nvox_z, 3)

        if not self.checked_imsize:
            self.proj_grid_voxel_ncam_indravel = np.zeros([*self.proj_grid_voxel_ncam.shape[:-1]], dtype='int64')
            for i in range(self.ncam):
                im = ims[i]
                assert len(im.shape)==3 and im.shape[2]==1
                assert im.data.c_contiguous
                np.clip(self.proj_grid_voxel_ncam[i, ..., 0], 0, im.shape[1] - 1, out=self.proj_grid_voxel_ncam[i, ..., 0])
                np.clip(self.proj_grid_voxel_ncam[i, ..., 1], 0, im.shape[0] - 1, out=self.proj_grid_voxel_ncam[i, ..., 1])

                indravel = np.ravel_multi_index(np.moveaxis(self.proj_grid_voxel_ncam[i, ..., ::-1], -1, 0), im.shape[:2])
                self.proj_grid_voxel_ncam_indravel[i] = indravel[...]

            self.checked_imsize = True

        # result = voxel_sampling(ims[...,0], self.proj_grid_voxel_ncam_indravel, com_range[[1,0,2]][:,0], (self.nvox, self.nvox, self.nvox)) #nview, nvox, nvox, nvox
        result = voxel_sampling_c_last(ims[...,0], self.proj_grid_voxel_ncam_indravel, com_range[[1,0,2]][:,0], (self.nvox, self.nvox, self.nvox)) #nvox, nvox, nvox, nview
        X = result.astype(np.float32) #(nvox, nvox, nvox, nview)
        assert self.norm_im
        # X = np.moveaxis(X, 0, -1) #(self.nvox, self.nvox, self.nvox, self.ncam)
        X = processing.preprocess_3d(X)
        
        y_3d = np.empty((self.nvox, self.nvox, self.nvox, 14), dtype="float32")

        return [X, xgrid_roi], y_3d



class DataGenerator_3Dconv_torch_video_canvas_multivoxel(DataGenerator_3Dconv_torch):
    def set_video(self, videopath, gpu, pkldata):
        assert os.path.exists(videopath), "Video file not found"
        assert self.batch_size == 1, "Batch size must be 1 for video data"

        self.vid = CanvasReaderPannelMask(videopath, pkldata, gpu=gpu)
        self.nframes = len(self.vid)
        out_numpy_shape=self.vid.out_numpy_shape
        self.vid.release()

        self.vidinfo = get_info(videopath)
        self.nframes = self.vidinfo.count
        self.vidnoblock = CanvasReaderPannelNoblock(videopath, pkldata, gpu=gpu, out_numpy_shape=out_numpy_shape)
        self.iframe =0

        self.nclass, self.ncam, self.image_hw = out_numpy_shape[0], out_numpy_shape[1], out_numpy_shape[2:4]
        self.batch_size = self.nclass
        self.n_channels_in = 1 #gray
        self.gpu = gpu
        self.device = torch.device(f"cuda:{gpu}")

        self.pkldata = pkldata
        coms_3d = pkldata['coms_3d']
        coms_3d = rectify_com3d_nan(coms_3d)
        coms_3d = coms_3d.reshape(-1, coms_3d.shape[-1])
        self.voxel_size_list = self.kwargs.get("vol_size_list", None)
        if self.voxel_size_list is None or self.voxel_size_list is []:
            self.voxel_size_list = [self.exp_voxel_size]*self.nclass
        else:
            assert len(self.voxel_size_list) == self.nclass

        self.init_grids(coms_3d)


    def release(self):
        self.vidnoblock.release()

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
        ret, im_pannels_nclass = self.vidnoblock.read()
        assert len(im_pannels_nclass) == self.batch_size == len(list_IDs_temp)
        assert not self.depth
        assert self.mode == "3dprob"
        assert self.mono

        X, xgrid_roi, y_3d = [], [], []
        for i, ID in enumerate(list_IDs_temp):
            experimentID = int(ID.split("_")[0])
            assert experimentID == 0
            ims = im_pannels_nclass[i]
            if 'coms_3d' in self.pkldata:
                com_3d = self.pkldata['coms_3d'][self.iframe,i,:]
            else:
                coms_2d = ims_to_com2ds(ims)
                com_3d = com2ds_to_com3d(coms_2d, self.ba_pose)

            [X_each, xgrid_roi_each], y_3d_each = self.quary_gridsample_by_com3d(com_3d, ims, i) #cxf
            X.append(X_each)
            xgrid_roi.append(xgrid_roi_each)
            y_3d.append(y_3d_each)
        self.iframe += 1
        X = np.stack(X, axis=0)
        return [X, xgrid_roi], y_3d

    def init_grids(self, com_3ds):
        self.grid_1d_l = []
        self.grid_coord_3d_l = []
        self.proj_grid_voxel_ncam_l = []
        self.proj_grid_voxel_ncam_indravel_l = []

        for voxel_size in self.voxel_size_list:
            self.vmin, self.vmax = -voxel_size / 2, voxel_size / 2
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
            assert len(camParams) == self.ncam
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
            
            self.proj_grid_voxel_ncam = np.array(proj_grid_voxel_ncam).astype('int16')  #(ncam, nvox_y, nvox_x, nvox_z, 2=hw)
            self.proj_grid_voxel_ncam_indravel = np.zeros([*self.proj_grid_voxel_ncam.shape[:-1]], dtype='int64') #(ncam, nvox_y, nvox_x, nvox_z)
            for i in range(self.ncam):
                np.clip(self.proj_grid_voxel_ncam[i, ..., 0], 0, self.image_hw[1] - 1, out=self.proj_grid_voxel_ncam[i, ..., 0])
                np.clip(self.proj_grid_voxel_ncam[i, ..., 1], 0, self.image_hw[0] - 1, out=self.proj_grid_voxel_ncam[i, ..., 1])
                indravel = np.ravel_multi_index(np.moveaxis(self.proj_grid_voxel_ncam[i, ..., ::-1], -1, 0), self.image_hw)
                self.proj_grid_voxel_ncam_indravel[i] = indravel[...]

            self.grid_1d = (xgrid, ygrid, zgrid)
            self.grid_coord_3d = np.stack([x_coord_3d, y_coord_3d, z_coord_3d], axis=-1).astype('float16')  #(nvox_y, nvox_x, nvox_z, 3)
            
            self.grid_1d_l.append(self.grid_1d)
            self.grid_coord_3d_l.append(self.grid_coord_3d)
            self.proj_grid_voxel_ncam_l.append(self.proj_grid_voxel_ncam)
            self.proj_grid_voxel_ncam_indravel_l.append(self.proj_grid_voxel_ncam_indravel)


    def quary_gridsample_by_com3d(self, com_3d:np.ndarray, ims:np.ndarray, iclass:int):
        # input=gray, output=gray.
        assert len(com_3d)==3
        assert len(ims) == self.ncam
        grid_coord_3d = self.grid_coord_3d_l[iclass]
        grid_1d = self.grid_1d_l[iclass]
        proj_grid_voxel_ncam_indravel = self.proj_grid_voxel_ncam_indravel_l[iclass]
        com_index = np.array([np.searchsorted(grid_1d[i], com_3d[i], side='right')
                        for i in range(3)])  #(3,)
        com_range = np.floor(com_index[:,None] + [- self.nvox/2, self.nvox/2]).astype(int) #(3,2)

        xgrid_roi = grid_coord_3d[ com_range[1][0]:com_range[1][1], 
                                   com_range[0][0]:com_range[0][1], 
                                   com_range[2][0]:com_range[2][1],
                                   :]   #(nvox_y, nvox_x, nvox_z, 3)

        result = voxel_sampling_c_last(ims[...,0], proj_grid_voxel_ncam_indravel, com_range[[1,0,2]][:,0], (self.nvox, self.nvox, self.nvox)) #nvox, nvox, nvox, nview
        X = result.astype(np.float32) #(nvox, nvox, nvox, nview)
        assert self.norm_im
        X = processing.preprocess_3d(X)
        y_3d = np.empty((self.nvox, self.nvox, self.nvox, 14), dtype="float32")

        return [X, xgrid_roi], y_3d
