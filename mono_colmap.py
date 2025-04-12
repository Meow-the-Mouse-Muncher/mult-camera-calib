# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import cv2
import os
import torch
from torch import Tensor
import torchvision
import torchvision.transforms.functional as F
from torchvision import transforms
# 自适应直方图均衡化  AHE






def dcn_warp(voxelgrid: Tensor, flow_x: Tensor, flow_y: Tensor):
    # voxelgrid: [bs,ts,H,W] | flow: [bs,ts,H,W]
    bs, ts, H, W = voxelgrid.shape
    flow = torch.stack([flow_y, flow_x], dim=2)  # [bs,ts,2,H,W]
    flow = flow.reshape(bs, ts * 2, H, W)  # [bs,ts*2,H,W]
    #  单位矩阵 保证了 只对一张图处理
    weight = torch.eye(ts, device=flow.device).double().reshape(ts, ts, 1, 1)  # 返回 ts 张图 对ts张图做处理
    # 单位卷积核
    return torchvision.ops.deform_conv2d(voxelgrid, flow, weight)  # [bs,ts,H,W]


def nonzero_mean(tensor: Tensor, dim: int, keepdim: bool = False, eps=1e-3):
    numel = torch.sum((tensor > eps).float(), dim=dim, keepdim=keepdim)
    value = torch.sum(tensor, dim=dim, keepdim=keepdim)
    return value / (numel + eps)


def read_images(directory_path, image_size):
    # 获取目录中所有图像文件的文件名
    image_files = [f for f in os.listdir(directory_path) if f.endswith('.jpg') or f.endswith('.png')]
    image_files = sorted(image_files,key=lambda x :  int(x.split('.')[0]))
    global N
    N = len(image_files)
    # 初始化矩阵
    images_matrix = np.zeros((len(image_files), *image_size), dtype=np.uint8)
    image_name_list=[]
    # 读取每张图像并存储到矩阵中
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(directory_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        images_matrix[i] = np.array(image)

    return images_matrix,image_name_list


def pix_move(width,height,quat,depth,K,T_x,T_y,T_z):

    T_matrix = np.stack([T_x,T_y,T_z],axis=1)

    T = T_matrix.reshape(-1, 3, 1)  # 116,3,1

    quat = np.array(quat)

    pix_x = np.arange(0,width)
    pix_y = np.arange(0,height)
    pix_x , pix_y = np.meshgrid(pix_x,pix_y)
    pix_x = pix_x.reshape(-1)
    pix_y = pix_y.reshape(-1)
    print(pix_x)
    ones = np.ones_like(pix_x)

    # 旋转矩阵 计算
    R_matrix = R.from_quat(quat).as_matrix()  # 116，3，3
    K_inv = np.linalg.inv(K)
    K = K[np.newaxis,:,:].repeat(R_matrix.shape[0],axis=0)
    K_inv = K_inv[np.newaxis,:,:].repeat(R_matrix.shape[0],axis=0)
    r_move = K@R_matrix
    r_move = r_move@K_inv - np.eye(3)  # 116，3，3   uref - u

    T_move = K@T/depth               # 116,3,1
    pix_place = np.stack([pix_x,pix_y,ones],axis=0)  # 3，200000
    pix_place = pix_place[np.newaxis, :, :]
    pix_place = np.repeat(pix_place,r_move.shape[0] , axis=0)   # 116,3，200000
    pix_move = r_move@pix_place 
    pix_move = pix_move + T_move  # 116,3，200000

    x_move_sheet = pix_move[:,0,:].reshape(quat.shape[0],height,width)   # u 
    y_move_sheet = pix_move[:,1,:].reshape(quat.shape[0],height,width)   # v
 
    return x_move_sheet,y_move_sheet

        



if __name__ == '__main__':
# 4703.863770,4750.683105,750.655371,491.003268
# 2000 1000 6472.86 6419.42 1000 500
# 7191.73748782809,7190.76441594966,897.056114098032,862.573591961378
# 3463.13471858099	0	0
# 0	3463.11625407825	0
# 347.866576798087	325.714566240859	1
    fx,fy,cx,cy= 3463.13471858099,3463.11625407825,347.866576798087, 325.714566240859
    K=np.array([[3463.13471858099,  0,   0],
                [0   ,3463.11625407825,  0],
                [347.866576798087,325.714566240859,1]]).T
    # K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    h, w = 600, 600
    d = 20
    N = 0
    var_threshold = 5000
    
    
    save_path =  os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))
    os.mkdir(save_path) if not os.path.exists(save_path) else None
    files_num = len(os.listdir(save_path))
    extrinsic_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'colmap','002'))
    image_name = './test'
    # 读取图像并存储到矩阵中
    image_matrix,imagepatch_name = read_images(image_name, image_size=(h, w))
    file_path = os.path.join(extrinsic_path, 'images.txt') 
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    Tx_list = []
    Ty_list = []
    Tz_list = []
    Qw_list = []
    Qx_list = []
    Qy_list = []
    Qz_list = []
    name_list =[]
    for line in lines:
        if '.jpg' in line or '.png'in line :
            data = line.strip().split()
            # 获取最后3个参数
            params = np.float64(data[-5:-2])
            Qw_list.append(np.float64(data[-9]))
            Qx_list.append(np.float64(data[-8]))
            Qy_list.append(np.float64(data[-7]))
            Qz_list.append(np.float64(data[-6]))
            Tx_list.append(params[0])
            Ty_list.append(params[1])
            Tz_list.append(params[2])

            name_list.append(data[-1].split('.')[0])

    name_list=np.array(list(map(int,name_list)))

    # 使用zip将多个列表组合成一个元组的列表  
    combined = list(zip(name_list,Tx_list, Ty_list, Tz_list,Qx_list,Qy_list,Qz_list,Qw_list))  
    
    # 使用sorted函数按name的值进行排序  
    sorted_combined = sorted(combined, key=lambda item: item[0]) 

    # 使用zip将排序后的元组列表解开成单独的列表  
    name_list,Tx_list, Ty_list, Tz_list,Qx_list,Qy_list,Qz_list,Qw_list = zip(*sorted_combined)
    print(name_list)
    Tx_list = list(Tx_list)
    Ty_list = list(Ty_list)
    Tz_list = list(Tz_list)
    Qx_list = list(Qx_list)
    Qy_list = list(Qy_list)
    Qz_list = list(Qz_list)
    Qw_list = list(Qw_list)


    image_matrix = torch.from_numpy(image_matrix)
    image_matrix = image_matrix.type(torch.float64)

    quat = torch.stack([torch.Tensor(Qx_list), torch.Tensor(Qy_list), torch.Tensor(Qz_list), torch.Tensor(Qw_list)], dim=1)


    # 输出矩阵的形状
    image_matrix = image_matrix.unsqueeze(dim=0)
    Tx_list= torch.Tensor(Tx_list)
    Ty_list= torch.Tensor(Ty_list)
    Tz_list= torch.Tensor(Tz_list)
    Tx_list= Tx_list.type(torch.float64)
    Ty_list= Ty_list.type(torch.float64)
    Tz_list= Tz_list.type(torch.float64)
    Tx_max = torch.max(Tx_list) - torch.min(Tx_list)
    Ty_max = torch.max(Ty_list) - torch.min(Ty_list)


    if Tx_max > Ty_max:
        d_min = (Tx_max * fx / 2) / (w * 1 / 2)
        d_max = (Tx_max * fx / 2) / (w * 1 / 20)
        print("x is max")
    else:
        d_min = (Ty_max * fy / 2) / (h * 1 / 2)
        d_max = (Ty_max * fy / 2) / (h * 1 / 20)
        print("y is max")
    print("d_max={}".format(d_max))

    search_range=d_max-d_min
    stop_threshold=0.1
    start_depth = d_min
    end_depth = d_max
    delta_depth = (d_max - d_min)*0.618
    
    for best_focus_depth in np.arange(10,100,1):
        print(f'best_focus_depth={best_focus_depth}')
        x_move , y_move =  pix_move(w,h,quat,best_focus_depth,K,Tx_list,Ty_list,Tz_list)
        x_move = torch.from_numpy(x_move)
        y_move = torch.from_numpy(y_move)
        x_move = x_move.unsqueeze(dim=0)
        y_move = y_move.unsqueeze(dim=0)
        x_move = x_move.expand(-1,-1,h,w)
        y_move = y_move.expand(-1,-1,h,w)
    
        refocus_images = dcn_warp(image_matrix, x_move , y_move)
    
        ref_image = nonzero_mean(refocus_images, dim=1, keepdim=True).squeeze()
        image = ref_image/ref_image.max()

   
        # new_image = np.zeros_like(image)
        # if image.shape[-1] == 3:
        #     new_image[:,:,0] = exposure.equalize_adapthist(image[:,:,0])
        #     new_image[:,:,1] = exposure.equalize_adapthist(image[:,:,1])
        #     new_image[:,:,2] = exposure.equalize_adapthist(image[:,:,2])
        # else:
        #     new_image = exposure.equalize_adapthist(image)
        new_image = np.uint8(np.clip(image*255., 0, 255))
        cv2.imwrite(f"{save_path}/{best_focus_depth:.3f}.png", new_image)
        print(f'OK,{image_name} is done!')

    