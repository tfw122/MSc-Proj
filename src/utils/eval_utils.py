import numpy as np
import pandas as pd
import os
import cv2
from collections import Counter
import pickle
from scipy.interpolate import LinearNDInterpolator
from glob import glob
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt 

'''
to get disparity from depth:  https://github.com/mrharicot/monodepth/issues/118

depth = baseline * focal / disparity

however model disparity output is at a different scale... therefore it needs to be rescaled to
the width of the original image

therefore:
depth = baseline * focal / (width * disparity)

for kitt this would be:  depth = 0.54 * 721 / (1242 * disp) For KITTI the baseline is 0.54m and the focal ~721 pixels. The relative disparity outputted by 
                                                            the model has to be scaled by 1242 which is the original image size.
disp_pp = cv2.resize(disp_pp, (original_width, original_height))

depth = 0.54 * 721/(1242*disp_pp)

'''




# adapted from: https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def post_process_disparity(l_disp, r_disp):

    (_, h, w) = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp
#----------------------------------------------- evaluation stuff --------------------------------------------------------
def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def compute_errors_scared(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)
    mean_abs= mean_absolute_error(gt, pred)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, mean_abs

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351
# -------------------------------------------------------------------------------------------------------
# The following code is for the small Kitti Dataset with GT Disparities provided, i.e. Kitti2015

def load_disp_kitti(path):
    all_disps= sorted(glob('{}/*.png'.format(path)))
    gt_disparities = []
    for i in range(len(all_disps)):
        disp = cv2.imread(all_disps[i], -1)
        disp = disp.astype(np.float32) / 256
        gt_disparities.append(disp)
    return np.asarray(gt_disparities)

def load_disp_scared(path_list):
    gt_disparities = []
    for i in range(len(path_list)):
        disp = cv2.imread(path_list[i], -1)
        disp = disp.astype(np.float32) / 256
        gt_disparities.append(disp)
    return gt_disparities

def convert_disps_to_depths_kitti(gt_disparities, pred_disparities):
    gt_depths = []
    pred_depths = []
    pred_disparities_resized = []
    
    for i in range(len(gt_disparities)):
        gt_disp = gt_disparities[i]
        height, width = gt_disp.shape

        pred_disp = pred_disparities[i]
        pred_disp = width * cv2.resize(pred_disp, (width, height), interpolation=cv2.INTER_LINEAR)

        pred_disparities_resized.append(pred_disp) 

        mask = gt_disp > 0

        gt_depth = width_to_focal[width] * 0.54 / (gt_disp + (1.0 - mask))
        pred_depth = width_to_focal[width] * 0.54 / pred_disp

        gt_depths.append(gt_depth)
        pred_depths.append(pred_depth)
    return gt_depths, pred_depths, pred_disparities_resized

def convert_disps_to_depths_scared(gt_disparities, pred_disparities, dataset_id):
    gt_depths = []
    pred_depths = []
    pred_disparities_resized = []
    
    if dataset_id==9:
        b= 4.132568439900515
        f= 1110.2546425040427
    elif dataset_id==8:
        b= 4.348277165573036
        f= 1112.709358269966

    for i in range(len(gt_disparities)):
        gt_disp = gt_disparities[i]
        height, width = gt_disp.shape
        
        if height != 1024:
            raise Exception('height not same as the original image height')
        if width != 1280:
            raise Exception('width not same as the original image width')

        pred_disp = pred_disparities[i] #* 2.0
        pred_disp = width * cv2.resize(pred_disp, (width, height), interpolation=cv2.INTER_LINEAR) #* 0.001

        pred_disparities_resized.append(pred_disp) 

        mask = gt_disp > 0

        gt_depth = f * (b/1000) / (gt_disp + (1.0 - mask))
        pred_depth = f * (b/1000) / pred_disp

        #print(np.max(gt_depth), np.min(gt_depth), np.max(pred_depth), np.min(pred_depth))
        gt_depths.append(gt_depth)
        pred_depths.append(pred_depth)
    return gt_depths, pred_depths, pred_disparities_resized

def convert_disps_to_depths_scared_mini(pred_disparities, dataset=9):
    pred_depths = []
    pred_disparities_resized = []
    # original size:
    width, height= 1280, 1024
    if dataset==9:
        b= 4.132568439900515
        f= 1110.2546425040427
    elif dataset==8:
        b= 4.348277165573036
        f= 1112.709358269966

    for i in range(len(pred_disparities)):
        
        pred_disp = pred_disparities[i]
        pred_disp = width * cv2.resize(pred_disp, (width, height), interpolation=cv2.INTER_LINEAR)

        pred_disparities_resized.append(pred_disp) 

        #mask = gt_disp > 0

        #gt_depth = width_to_focal[width] * 0.54 / (gt_disp + (1.0 - mask))
        pred_depth = f * b / pred_disp

        #gt_depths.append(gt_depth)
        pred_depths.append(pred_depth)
    return pred_depths, pred_disparities_resized

# -------------------------------------------------------------------------------------------------------
# The following code is for the raw Kitti Dataset withVelodyne points (raw LiDAR points)

def load_velodyne_points(file_name):
    # adapted from https://github.com/hunse/kitti
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points

def get_xyd(depth_map):
    # get the location of x,y for valid pixels.
    x,y= np.where(depth_map > 0)
    # get depth values for valid pixels only;
    d= depth_map[depth_map!=0]
    # generate an Nx3 array;
    xyd= np.stack((y,x,d)).T 
    return xyd
    
# used for smoothing:
def lin_interp(shape, depth_map):
    # taken from https://github.com/hunse/kitti
    from scipy.interpolate import LinearNDInterpolator
    xyd= get_xyd(depth_map)
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity

def read_file_data(files, data_root):
    gt_files = []
    gt_calib = []
    im_sizes = []
    im_files = []
    cams = []
    num_probs = 0
    for filename in files:
        filename = filename.split()[0]
        splits = filename.split('/')
        camera_id = np.int32(splits[2][-1:])   # 2 is left, 3 is right
        date = splits[0]
        im_id = splits[4][:10]
        file_root = '{}/{}'
        
        im = filename
        vel = '{}/{}/velodyne_points/data/{}.bin'.format(splits[0], splits[1], im_id)

        if os.path.isfile(data_root + im):
            gt_files.append(data_root + vel)
            gt_calib.append(data_root + date + '/')
            im_sizes.append(cv2.imread(data_root + im).shape[:2])
            im_files.append(data_root + im)
            cams.append(2)
        else:
            num_probs += 1
            print('{} missing'.format(data_root + im))
    print(num_probs, 'files missing')

    return gt_files, gt_calib, im_sizes, im_files, cams


def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def get_focal_length_baseline(calib_dir, cam):
    cam2cam = read_calib_file(calib_dir + 'calib_cam_to_cam.txt')
    P2_rect = cam2cam['P_rect_02'].reshape(3,4)
    P3_rect = cam2cam['P_rect_03'].reshape(3,4)

    # cam 2 is left of camera 0  -6cm
    # cam 3 is to the right  +54cm
    b2 = P2_rect[0,3] / -P2_rect[0,0]
    b3 = P3_rect[0,3] / -P3_rect[0,0]
    baseline = b3-b2

    if cam==2:
        focal_length = P2_rect[0,0]
    elif cam==3:
        focal_length = P3_rect[0,0]

    return focal_length, baseline


def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1

def generate_depth_map(calib_dir, velo_file_name, im_shape, cam=2, interp=False, vel_depth=False):
    # load calibration files
    cam2cam = read_calib_file(calib_dir + 'calib_cam_to_cam.txt')
    velo2cam = read_calib_file(calib_dir + 'calib_velo_to_cam.txt')
    velo2cam = np.hstack((velo2cam['R'].reshape(3,3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3,:3] = cam2cam['R_rect_00'].reshape(3,3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3,4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_file_name)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:,:2] / velo_pts_im[:,2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:,0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:,1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:,0] < im_shape[1]) & (velo_pts_im[:,1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds==dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth<0] = 0

    if interp:
        # interpolate the depth map to fill in holes
        depth_interp = lin_interp(im_shape, velo_pts_im)
        return depth, depth_interp
    else:
        return depth