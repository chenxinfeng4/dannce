# from lilab.multiview_scripts_dev.s4_matpkl2matcalibpkl import build_input_short, project_points_short
from multiview_calib.calibpkl_predict import CalibPredict
from scipy.ndimage.measurements import center_of_mass
import numpy as np

def ims_to_com2ds_orig(ims):
    ims_mask = [(im[:,:,0]>1)+0.01 for im in ims]
    coms_2d = [center_of_mass(im_mask)[::-1] for im_mask in ims_mask]
    return coms_2d

def ims_to_com2ds(ims):
    coms_2d = []
    for im_mask in ims:
        if im_mask.ndim == 3:
            im_mask = im_mask[:,:,0]
        com_2d = center_of_mass(im_mask)[::-1] if np.max(im_mask) > 10 else np.ones((2,))+np.nan
        coms_2d .append(com_2d)
    coms_2d = np.array(coms_2d)
    return coms_2d

def matlab_pose_to_cv2_pose(camParamsOrig):
    keys = ['K', 'RDistort', 'TDistort', 't', 'r']
    camParams = list()
    if type(camParamsOrig) is np.ndarray:
        camParamsOrig = np.squeeze(camParamsOrig)
        for icam in range(len(camParamsOrig)):
            camParam = {key: camParamsOrig[icam][0][key][0] for key in keys}
            if 'R' not in camParam: camParam['R'] = camParam['r']
            camParams.append(camParam)
    else:
        assert len(camParamsOrig) > 2
        for camName, camParamOrg in camParamsOrig.items():
            if 'R' not in camParamOrg: camParamOrg['R'] = camParamOrg['r']
            camParams.append(camParamOrg)

    # from matlab to opencv
    poses = dict()
    for icam in range(len(camParams)):
        rd = camParams[icam]['RDistort'].reshape((-1))
        td = camParams[icam]['TDistort'].reshape((-1))
        dist = np.zeros((8,))
        dist[:2] = rd[:2]
        dist[2:4] = td[:2]
        dist[3]  = rd[2]
        poses[icam] = {'R': camParams[icam]['R'].T, 
                    't': camParams[icam]['t'].reshape((3,)),
                    'K': camParams[icam]['K'].T - [[0, 0, 1], [0,0,1], [0,0,0]],
                    'dist':dist}
    return poses


def com2ds_to_com3d(com2ds, poses):
    com2ds = np.array(com2ds)
    calibPredict = CalibPredict({'ba_poses': poses})
    com_3d = calibPredict.p2d_to_p3d(com2ds)
    com_3d = np.squeeze(com_3d)
    return com_3d


def com3d_to_com2ds(com_3d, poses):
    calibPredict = CalibPredict({'ba_poses': poses})
    com_2ds = calibPredict.p3d_to_p2d(com_3d)
    return com_2ds