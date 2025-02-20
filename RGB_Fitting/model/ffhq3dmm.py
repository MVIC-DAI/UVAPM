import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
from PIL import Image


class FFHQ3DMM(nn.Module):
    def __init__(self, 
                 model_folder,
                 model_resolution=64,
                 device='cuda'):
        super(FFHQ3DMM, self).__init__()
        self.model_resolution = model_resolution 

        model_dict_path = os.path.join(model_folder, f"model_{model_resolution}.pkl")
        assert os.path.exists(model_dict_path)
        with open(model_dict_path, 'rb') as f:
            model_dict = pickle.load(f)
        self.r_mean = model_dict['r']['mean'].astype(np.float32).T   # shape:[1, 64*64]  
        self.r_base = model_dict['r']['base'].astype(np.float32)   # shape:[64*64, 100]  
        self.r_dims = self.r_base.shape[1]  # 100
        self.g_mean = model_dict['g']['mean'].astype(np.float32).T   # shape:[1, 64*64]  
        self.g_base = model_dict['g']['base'].astype(np.float32)   # shape:[64*64, 100]  
        self.g_dims = self.g_base.shape[1]  # 100
        self.b_mean = model_dict['b']['mean'].astype(np.float32).T   # shape:[1, 64*64]  
        self.b_base = model_dict['b']['base'].astype(np.float32)   # shape:[64*64, 100]  
        self.b_dims = self.b_base.shape[1]  # 100
        
        self.np2tensor()
        self.to(device)
        self.device = device
    
    def compute_uv_texture(self, coeffs, out_resolution=64, normalize=True):
        '''
        compute uv texture map from coefficient

        Args:
            coeffs : torch.Tensor, (B, 300). The texture coeffs. 
                            RGB:R:0-99 G:100-199 B:200-299
            out_resolution (int, optional): the out resolution of output uv map. Defaults to 64*64.
            
        Returns:
            _type_: _description_
        '''
        assert coeffs.shape[-1]==self.r_dims+self.g_dims+self.b_dims
        coeffs_dict = self.split_coeff(coeffs)
        batch_size = coeffs.shape[0]
        r_part = torch.einsum('ij,aj->ai', self.r_base, coeffs_dict['r'])   #[B, 64*64]
        g_part = torch.einsum('ij,aj->ai', self.g_base, coeffs_dict['g'])   #[B, 64*64]
        b_part = torch.einsum('ij,aj->ai', self.b_base, coeffs_dict['b'])   #[B, 64*64]

        face_texture_r = (r_part + self.r_mean).reshape([batch_size, self.model_resolution, self.model_resolution]) #[B, 64, 64]
        face_texture_g = (g_part + self.g_mean).reshape([batch_size, self.model_resolution, self.model_resolution]) #[B, 64, 64]
        face_texture_b = (b_part + self.b_mean).reshape([batch_size, self.model_resolution, self.model_resolution]) #[B, 64, 64]
        
        face_texture = torch.stack((face_texture_r, face_texture_g, face_texture_b), dim=-1)    #[B, 64, 64, 3]
        face_texture = face_texture.permute(0, 3, 1, 2)      #[B, 3, 64, 64]
        
        # if out_resolution>self.model_resolution:
        #     # 上采样
        #     face_texture = F.interpolate(face_texture, size=(out_resolution, out_resolution), mode='bilinear', align_corners=False)
        face_texture = F.interpolate(face_texture, size=(out_resolution, out_resolution), mode='bilinear', align_corners=False)

        if normalize:
            face_texture = face_texture / 255.

        return face_texture
    
    def split_coeff(self, coeffs):
        '''
        Split the estimated coeffs.
        '''
        r_coeffs = coeffs[:, :self.r_dims]
        g_coeffs = coeffs[:, self.r_dims:self.r_dims + self.g_dims]
        b_coeffs = coeffs[:, self.r_dims + self.g_dims:self.r_dims + self.g_dims + self.b_dims]

        return {
            'r': r_coeffs, # 100
            'g': g_coeffs, # 100
            'b': b_coeffs, # 100
        }

    # def combine_coeff(self, coeffs_dict):
    #     '''
    #     Combine the estimated coeffs.
    #     '''
    #     coeffs = torch.cat([
    #         coeffs_dict['r'],
    #         coeffs_dict['g'],
    #         coeffs_dict['b'],
    #     ], dim=1)
    #     return coeffs
    
    def np2tensor(self):
        '''
        Transfer numpy.array to torch.Tensor.
        '''
        for key, value in self.__dict__.items():
            if type(value).__module__ == np.__name__:
                setattr(self, key, torch.tensor(value))

    def tensor2np(self):
        '''
        Transfer torch.Tensor to numpy.array.
        '''
        for key, value in self.__dict__.items():
            if type(value).__module__ == torch.__name__:
                setattr(self, key, value.detach().cpu().numpy())

    def to(self, device):
        '''
        Move to device.
        '''
        self.device = device
        for key, value in self.__dict__.items():
            if type(value).__module__ == torch.__name__:
                setattr(self, key, value.to(device))

# import os
# import torch
if __name__ == '__main__':
    # 确定模型数据的文件夹路径（请根据实际情况修改）
    model_folder = "/media/wang/SSD_2/demo/VIFR/data/uvmap3dmm"  # 修改为您实际的模型文件夹路径

    # 检查模型文件是否存在
    model_resolution = 128
    model_dict_path = os.path.join(model_folder, f"model_{model_resolution}.pkl")
    assert os.path.exists(model_dict_path), "模型文件不存在，请检查路径!"

    # 1. 创建FFHQ3DMM对象
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    face_model = FFHQ3DMM(model_folder=model_folder, model_resolution=model_resolution, device=device)

    # 2. 生成一些随机的系数数据
    batch_size = 2  # 一次处理2个样本
    # coeffs = torch.randn(batch_size, 300).to(device)*20  # 生成随机系数


    coeffs = torch.zeros(batch_size, 300, dtype=torch.float32, device=device)
    coeffs[0, 1] = 100.0
    coeffs[0, 101] = 200.0
    coeffs[0, 201] = 100.0
    coeffs[0, 202] = 300.0
    # coeffs[0, 2] = 0.0
    # coeffs[0, 3] = 5.0

    # 3. 计算UV纹理图
    uv_texture = face_model.compute_uv_texture(coeffs, out_resolution=1024, normalize=True)

    # 4. 查看输出
    print(f"生成的UV纹理形状: {uv_texture.shape}")
    # 输出 uv_texture 的形状，应该是 [B, 3, 64, 64]
    print(f"UV纹理数据示例: {uv_texture[0]}")  # 打印第一张纹理的值

    # 将输出转换为 NumPy 数组以便于展示和保存
    output_numpy = uv_texture.permute(0, 2, 3, 1).detach().cpu().numpy()

    # 保存第一张图像
    first_image = (output_numpy[0] * 255).astype(np.uint8)  # 将值调整到[0, 255]并转换为uint8类型
    first_image_pil = Image.fromarray(first_image)  # 使用PIL将NumPy数组转换为图像
    first_image_pil.save("first_image.png")  # 保存图像为PNG格式