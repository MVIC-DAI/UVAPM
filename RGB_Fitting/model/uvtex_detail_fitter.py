import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import cv2

from .optimizers import SphericalOptimizer
from .losses import perceptual_loss, photo_loss, vgg_loss, latents_geocross_loss
from utils.preprocess_utils import estimate_norm_torch
from utils.data_utils import tensor2np, draw_mask, draw_landmarks, img3channel, hsv_to_rgb, rgb_to_hsv
from .ffhq3dmm import FFHQ3DMM




import torch
import torch.nn as nn
from torchvision import models

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # 使用预训练的 ResNet50 作为编码器
        resnet = models.resnet50(pretrained=True)
        # 修改输入层以接受单通道图像
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # 去掉最后的全连接层
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # 只保留卷积层部分
        self.encoder.add_module("flatten", nn.Flatten())  # 添加 Flatten 层
        
        # 潜在空间的均值和方差
        self.fc_mu = nn.Linear(2048, 500)  # 均值
        self.fc_logvar = nn.Linear(2048, 500)  # 对数方差

        # 解码器部分
        self.decoder = nn.Sequential(
            nn.Linear(500, 512),  # 输入维度改为 384
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512 * 512),  # 输出为 512*512 的图像
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 512, 512))  # 重塑为 512x512 的图像
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(2.0 * logvar)  # 计算标准差 0.5
        eps = torch.randn_like(std)  # 从标准正态分布中采样
        return mu + eps * std  # 重参数化

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)  # 计算均值
        logvar = self.fc_logvar(x)  # 计算对数方差
        z = self.reparameterize(mu, logvar)  # 重参数化
        x_reconstructed = self.decoder(z)  # 解码
        return x_reconstructed, mu, logvar  # 返回重构的图像、均值和对数方差

class Decoder(nn.Module):
    def __init__(self, vae):
        super(Decoder, self).__init__()
        self.decoder = vae.decoder  # Extract the decoder from the trained VAE

    def forward(self, z):
        return self.decoder(z)  # Forward pass through the decoder

# Load the trained VAE model
def load_trained_vae(model_path):
    vae = VAE()  # Initialize the VAE model
    vae.load_state_dict(torch.load(model_path))  # Load the trained weights
    vae.eval()  # Set the model to evaluation mode
    return vae


# import torch
# import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, out_channels=1, out_scale=0.01, sample_mode='bilinear'):
        super(Generator, self).__init__()
        self.out_scale = out_scale
        
        self.init_size = 32 // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode=sample_mode),  # 16
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode),  # 32
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode),  # 64
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode),  # 128
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode),  # 256
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode),  # 512
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(8, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode),  # 1024
            nn.Conv2d(8, out_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img * self.out_scale



class Fitter:

    def __init__(self,
                 facemodel,
                 uv3dmm_model,
                 uv_detail_model,
                 renderer,
                 net_recog,
                 net_vgg,
                 logger,
                 input_data,
                 init_coeffs,
                 init_latents_z=None,
                 init_detail_coeffs=None,
                 **kwargs):
        # parametric face model
        self.facemodel = facemodel
        # uv_3dmm
        self.uv3dmm_model = uv3dmm_model
        # uv detail mdoel
        # self.uv_detail_model = uv_detail_model.eval().requires_grad_(False)

        trained_vae = load_trained_vae('/root/autodl-tmp/demo/FFHQ-UV/checkpoints/detail_model/variational_autoencoder.pth')
        self.uv_detail_model = Decoder(trained_vae).to('cuda').eval()

        # renderer
        self.renderer = renderer
        # the recognition model
        self.net_recog = net_recog.eval().requires_grad_(False)
        # the vgg model
        self.net_vgg = net_vgg.eval()

        # set fitting args
        self.w_feat = kwargs['w_feat'] if 'w_feat' in kwargs else 0.
        self.w_color = kwargs['w_color'] if 'w_color' in kwargs else 0.
        self.w_vgg = kwargs['w_vgg'] if 'w_vgg' in kwargs else 0.
        self.w_reg_latent = kwargs['w_reg_latent'] if 'w_reg_latent' in kwargs else 0.
        self.initial_lr = kwargs['initial_lr'] if 'initial_lr' in kwargs else 0.001
        self.lr_rampdown_length = kwargs['lr_rampdown_length'] if 'lr_rampdown_length' in kwargs else 0.25
        self.total_step = kwargs['total_step'] if 'total_step' in kwargs else 100
        self.print_freq = kwargs['print_freq'] if 'print_freq' in kwargs else 10
        self.visual_freq = kwargs['visual_freq'] if 'visual_freq' in kwargs else 10

        # input data for supervision
        self.input_img = input_data['img']
        self.skin_mask = input_data['skin_mask']
        self.parse_mask = input_data['parse_mask']
        self.gt_lm = input_data['lm']
        self.trans_m = input_data['M']
        with torch.no_grad():
            recog_output = self.net_recog(self.input_img, self.trans_m)
        self.input_img_feat = recog_output

        # init coeffs
        self.coeffs_opt = init_coeffs
        self.coeffs_opt.requires_grad = False  # fix shape
        self.coeffs_opt_dict = self.facemodel.split_coeff(self.coeffs_opt)

        # init z coeffs
        self.latents_z = init_latents_z
        # print(self.latents_z.shape)
        self.latents_z.requires_grad = False
        # print(self.uv3dmm_model(self.latents_z).shape, "66")
        corase = F.interpolate(self.uv3dmm_model(self.latents_z), size=(512, 512), mode='bilinear', align_corners=False)

        self.coarse_uv_map = rgb_to_hsv(corase).cuda()
        print(f"self.coarse_uv_map.shape:{self.coarse_uv_map.shape}")
        print(f"self.coarse_uv_map.max:{torch.max(self.coarse_uv_map[:,2,:,:])},self.coarse_uv_map.min:{torch.min(self.coarse_uv_map[:,2,:,:])}")
        # coarse_uv_map = coarse_uv_map.squeeze(0).permute(1, 2, 0).cpu().numpy().astype('uint8')  # 转换为HWC格式
        # coarse_uv_map = cv2.cvtColor(coarse_uv_map, cv2.COLOR_RGB2HSV)
        # self.coarse_uv_map = torch.from_numpy(coarse_uv_map).permute(2, 0, 1).unsqueeze(0).cuda()

        # # 将RGB图像从[0, 1]范围转换为[0, 255]范围
        # coarse_uv_map = (coarse_uv_map)
        
        if init_detail_coeffs is not None:
            self.latents_detail_opt = init_detail_coeffs
            self.latents_detail_opt.requires_grad = True
        else:
            self.latents_detail_opt = torch.zeros((1, 500), requires_grad=True, device="cuda")
        # coeffs = 
        self.optimizer = torch.optim.Adam([self.latents_detail_opt], lr=1.0)

        self.initial_lr_list = [self.initial_lr]
        self.now_step = 0

        # logger
        self.logger = logger

    def update_learning_rate(self):
        t = float(self.now_step) / self.total_step
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        for i, param_group in enumerate(self.optimizer.opt.param_groups):
            lr = self.initial_lr_list[i] * lr_ramp
            param_group['lr'] = lr

    def forward(self):
        # forward face model
        self.pred_vertex, self.pred_tex, self.pred_shading, self.pred_color, self.pred_lm = \
            self.facemodel.compute_for_render(self.coeffs_opt_dict)
        
        # self.detail_uv_map = self.uv_detail_model(self.latents_detail_opt).squeeze(0)# torch.Size([1, 1, 224, 224])
        # print(self.detail_uv_map.shape, "1")
        # print(self.pred_uv_map.shape, "2")
        # self.detail_uv_map = F.interpolate(self.detail_uv_map, size=(224, 224), mode='bilinear', align_corners=True) # torch.Size([1, 1, 224, 224])
        detail_v = self.uv_detail_model(self.latents_detail_opt)*1e-2
        # print(f"detail_v.shape:{detail_v.shape}")
        print(f"detail_v.max:{torch.max(detail_v[:,0,:,:])},detail_v.min:{torch.min(detail_v[:,0,:,:])}")
        detail_uv_map = self.coarse_uv_map.clone()
        detail_uv_map[:,2,:,:] = detail_v[:,0,:,:]
        # self.pred_uv_map = torch.stack((self.coarse_uv_map[:,0,:,:], self.coarse_uv_map[:,1,:,:], detail_v[:,0,:,:]), dim=1)
        # print(f"self.pred_uv_map.shape:{self.pred_uv_map.shape}")
        self.pred_uv_map = hsv_to_rgb(detail_uv_map)

        # render full head
        vertex_uv_coord = self.facemodel.vtx_vt.unsqueeze(0).repeat(self.latents_detail_opt.size()[0], 1, 1)
        render_feat = torch.cat([vertex_uv_coord, self.pred_shading], axis=2)
        self.render_head_mask, _, self.render_head = \
            self.renderer(self.pred_vertex, self.facemodel.head_buf, feat=render_feat, uv_map=self.pred_uv_map)
        # render front face
        self.render_face_mask, _, self.render_face = \
            self.renderer(self.pred_vertex, self.facemodel.face_buf, feat=render_feat, uv_map=self.pred_uv_map)

    def compute_losses(self):
        # initial loss
        self.loss_names = ['all']
        self.loss_all = 0.
        # inset front face with input image
        # render_face_mask = self.render_face_mask.detach()
        render_face_mask = self.render_face_mask
        render_face = self.render_face * render_face_mask + (1 - render_face_mask) * self.input_img
        # id feature loss
        if self.w_feat > 0:
            assert self.net_recog.training == False
            if self.pred_lm.shape[1] == 68:
                pred_trans_m = estimate_norm_torch(self.pred_lm, self.input_img.shape[-2])
            else:
                pred_trans_m = self.trans_m
            pred_feat = self.net_recog(render_face, pred_trans_m)
            self.loss_feat = perceptual_loss(pred_feat, self.input_img_feat)
            self.loss_all += self.w_feat * self.loss_feat
            self.loss_names.append('feat')
        # color loss
        if self.w_color > 0:
            loss_face_mask = render_face_mask * self.parse_mask * self.skin_mask
            self.loss_color = photo_loss(render_face, self.input_img, loss_face_mask)
            self.loss_all += self.w_color * self.loss_color
            self.loss_names.append('color')
        # vgg loss, using the same render_face(face_mask) with color loss
        if self.w_vgg > 0:
            loss_face_mask = render_face_mask * self.parse_mask
            render_face_vgg = render_face * loss_face_mask
            input_face_vgg = self.input_img * loss_face_mask
            self.loss_vgg = vgg_loss(render_face_vgg, input_face_vgg, self.net_vgg)
            self.loss_all += self.w_vgg * self.loss_vgg
            self.loss_names.append('vgg')
        # w latent geocross regression
        if self.w_reg_latent > 0:
            # reg_r_loss, reg_g_loss, reg_b_loss = reg_loss_fn(coeffs=self.latents_z_opt, split=True)
            # self.loss_reg_latent = reg_r_loss + reg_g_loss + reg_b_loss
            # self.loss_all += self.loss_reg_latent * 1e-7
            self.loss_reg_latent = 0.0
            self.loss_all += self.loss_reg_latent * 0.0
            self.loss_names.append('reg_latent')

    def optimize_parameters(self):
        # self.update_learning_rate()
        self.forward()
        self.compute_losses()
        self.optimizer.zero_grad()
        self.loss_all.backward()
        # self.loss_all.backward(retain_graph=True)
        self.optimizer.step()
        self.now_step += 1

    def gather_visual_img(self):
        # input data
        input_img = tensor2np(self.input_img[:1, :, :, :])
        skin_img = img3channel(tensor2np(self.skin_mask[:1, :, :, :]))
        parse_mask = tensor2np(self.parse_mask[:1, :, :, :], dst_range=1.0)
        gt_lm = self.gt_lm[0, :, :].detach().cpu().numpy()
        # predict data
        pre_uv_img = tensor2np(F.interpolate(self.pred_uv_map, size=input_img.shape[:2], mode='area')[:1, :, :, :])
        pred_face_img = self.render_face * self.render_face_mask + (1 - self.render_face_mask) * self.input_img
        pred_face_img = tensor2np(pred_face_img[:1, :, :, :])
        pred_head_img = tensor2np(self.render_head[:1, :, :, :])
        pred_lm = self.pred_lm[0, :, :].detach().cpu().numpy()
        # draw mask and landmarks
        parse_img = draw_mask(input_img, parse_mask)
        gt_lm[..., 1] = pred_face_img.shape[0] - 1 - gt_lm[..., 1]
        pred_lm[..., 1] = pred_face_img.shape[0] - 1 - pred_lm[..., 1]
        lm_img = draw_landmarks(pred_face_img, gt_lm, color='b')
        lm_img = draw_landmarks(lm_img, pred_lm, color='r')
        # combine visual images
        combine_img = np.concatenate([input_img, skin_img, parse_img, lm_img, pred_face_img, pred_head_img, pre_uv_img],
                                     axis=1)
        return combine_img

    def gather_loss_log_str(self):
        loss_log = {}
        loss_str = ''
        for name in self.loss_names:
            loss_value = float(getattr(self, 'loss_' + name))
            loss_log[f'loss/{name}'] = loss_value
            loss_str += f'[loss/{name}: {loss_value:.5f}]'
        return loss_log, loss_str

    def iterate(self):
        for _ in range(self.total_step):
            # optimize
            self.optimize_parameters()
            # print log
            if self.now_step % self.print_freq == 0 or self.now_step == self.total_step:
                loss_log, loss_str = self.gather_loss_log_str()
                now_lr = self.optimizer.param_groups[0]['lr']
                self.logger.write_tb_scalar(['lr'], [now_lr], self.now_step)
                self.logger.write_tb_scalar(loss_log.keys(), loss_log.values(), self.now_step)
                self.logger.write_txt_log(f'[step {self.now_step}/{self.total_step}] [lr:{now_lr:.7f}] {loss_str}')
            # save intermediate visual results
            if self.now_step % self.visual_freq == 0 or self.now_step == self.total_step:
                vis_img = self.gather_visual_img()
                self.logger.write_tb_images([vis_img], ['vis'], self.now_step)

        final_coeffs = self.coeffs_opt.detach().clone()
        final_latents_z = self.latents_z.detach().clone()
        # final_latents_w = self.latents_w_opt.detach().clone()
        # final_latents_w = final_latents_z
        final_detail_latents = self.latents_detail_opt.detach().clone()
        return final_coeffs, final_latents_z, final_detail_latents
