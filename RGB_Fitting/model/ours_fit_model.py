import os
import torch
import numpy as np
import torch.nn.functional as F
# import torch.nn as nn

from .hifi3dpp import ParametricFaceModel
from .renderer_nvdiffrast import MeshRenderer
from . import uvtex_spherical_fixshape_fitter, uvtex_wspace_shape_joint_fitter, uvtex_detail_fitter
from network import texgan
from network.recog import define_net_recog
from network.recon_deep3d import define_net_recon_deep3d
from network.stylegan2 import dnnlib
from utils.data_utils import setup_seed, tensor2np, draw_mask, draw_landmarks, img3channel
from utils.mesh_utils import unwrap_vertex_to_uv, write_mesh_obj, write_mtl
from .uvtex_spherical_fixshape_fitter import TextureGenerationModel
from .uvtex_detail_fitter import *


trained_vae = load_trained_vae('/root/autodl-tmp/demo/FFHQ-UV/checkpoints/detail_model/variational_autoencoder.pth')



class FitModel:

    def __init__(self, cpk_dir, topo_dir, uv3dmm_model_dir, loose_tex=False, lm86=False, device='cuda'):
        self.args_model = {
            # face model and renderer
            'fm_model_file': os.path.join(topo_dir, 'hifi3dpp_model_info.mat'),
            'unwrap_info_file': os.path.join(topo_dir, 'unwrap_1024_info.mat'),
            'camera_distance': 10.,
            'focal': 1015.,
            'center': 112.,
            'znear': 5.,
            'zfar': 15.,
            # texture gan
            # 'texgan_model_file': os.path.join(cpk_dir, f'texgan_model/{texgan_model_name}'),
            # uv3dmm model
            'uv3dmm_model_dir': uv3dmm_model_dir,
            # deep3d nn inference model
            'net_recon': 'resnet50',
            'net_recon_path': os.path.join(cpk_dir, 'deep3d_model/epoch_latest.pth'),
            # recognition model
            'net_recog': 'r50',
            'net_recog_path': os.path.join(cpk_dir, 'arcface_model/ms1mv3_arcface_r50_fp16_backbone.pth'),
            # vgg model
            'net_vgg_path': os.path.join(cpk_dir, 'vgg_model/vgg16.pt'),
        }
        self.args_s2_search_uvtex_spherical_fixshape = {
            'w_feat': 10.0,
            'w_color': 10.0,
            'w_vgg': 100.0,
            'w_reg_latent': 0.05,
            'initial_lr': 2.0,
            'lr_rampdown_length': 0.25,
            'total_step': 100, # 5000(paper)->3000
            'print_freq': 100, # 50
            'visual_freq': 10,
        }
        self.args_s3_predict_detail_uvtex = {
            'w_feat': 10.0,
            'w_color': 10.0,
            'w_vgg': 100.0,
            'w_reg_latent': 0.05,
            'initial_lr': 0.01,
            'lr_rampdown_length': 0.25,
            'total_step': 100, # 4000
            'print_freq': 50, # 50
            'visual_freq': 10,
        }

        self.args_names = ['model', 's2_search_uvtex_spherical_fixshape', 's3_predict_detail_uvtex']

        # parametric face model
        self.facemodel = ParametricFaceModel(fm_model_file=self.args_model['fm_model_file'],
                                             unwrap_info_file=self.args_model['unwrap_info_file'],
                                             camera_distance=self.args_model['camera_distance'],
                                             focal=self.args_model['focal'],
                                             center=self.args_model['center'],
                                             lm86=lm86,
                                             device=device)

        # texture gan
        # self.tex_gan = texgan.TextureGAN(model_path=self.args_model['texgan_model_file'], device=device)

        # uv3dmm model
        self.uv3dmm_model = TextureGenerationModel(model_folder=self.args_model['uv3dmm_model_dir'], model_resolution=256).to(device)
       
        # uv detail Generator
        # self.uv_detail_model = uvtex_detail_fitter.Generator(latent_dim=100, out_channels=1, out_scale=0.01, sample_mode = 'bilinear').to("cuda")
        self.uv_detail_model = Decoder(trained_vae).to('cuda').eval()
        
        # deep3d nn reconstruction model
        fc_info = {
            'id_dims': self.facemodel.id_dims,
            'exp_dims': self.facemodel.exp_dims,
            'tex_dims': self.facemodel.tex_dims
        }
        print(fc_info)
        self.net_recon_deep3d = define_net_recon_deep3d(net_recon=self.args_model['net_recon'],
                                                        use_last_fc=False,
                                                        fc_dim_dict=fc_info,
                                                        pretrained_path=self.args_model['net_recon_path'])
        self.net_recon_deep3d = self.net_recon_deep3d.eval().requires_grad_(False)

        # renderer
        fov = 2 * np.arctan(self.args_model['center'] / self.args_model['focal']) * 180 / np.pi
        self.renderer = MeshRenderer(fov=fov,
                                     znear=self.args_model['znear'],
                                     zfar=self.args_model['zfar'],
                                     rasterize_size=int(2 * self.args_model['center']))

        # the recognition model
        self.net_recog = define_net_recog(net_recog=self.args_model['net_recog'],
                                          pretrained_path=self.args_model['net_recog_path'])
        self.net_recog = self.net_recog.eval().requires_grad_(False)

        # the vgg model
        with dnnlib.util.open_url(self.args_model['net_vgg_path']) as f:
            self.net_vgg = torch.jit.load(f).eval()

        # coeffs and latents
        self.pred_coeffs = None
        self.pred_latents_w = None
        self.pred_latents_z = None
        self.pred_detail_latents = None

        self.to(device)
        self.device = device

    def to(self, device):
        self.device = device
        self.facemodel.to(device)
        # self.tex_gan.to(device)
        self.uv3dmm_model.to(device)
        self.uv_detail_model.to(device)
        self.net_recon_deep3d.to(device)
        self.renderer.to(device)
        self.net_recog.to(device)
        self.net_vgg.to(device)

    def infer_render(self, is_uv_tex=True):
        # forward face model
        self.pred_coeffs_dict = self.facemodel.split_coeff(self.pred_coeffs)
        self.pred_vertex, self.pred_tex, self.pred_shading, self.pred_color, self.pred_lm = \
            self.facemodel.compute_for_render(self.pred_coeffs_dict)
        if is_uv_tex:
            # forward texture gan
            # self.pred_uv_map = self.tex_gan.synth_uv_map(self.pred_latents_w)
            # model_folder = "/root/autodl-tmp/demo/FFHQ-UV/RGB_Fitting/model/uv3dmm/"
            # uv3dmm_model = TextureGenerationModel(model_folder=model_folder, model_resolution=256).to("cuda")
            # TextureGenerationModel(model_folder=model_folder, model_resolution=256).to("cuda")
            self.pred_uv_map = self.uv3dmm_model(self.pred_latents_z)
            if self.pred_detail_latents is not None:
                print("添加细节")
                corase = F.interpolate(self.pred_uv_map, size=(512, 512), mode='bilinear', align_corners=False)
                coarse_uv_map = rgb_to_hsv(corase).cuda()
                coarse_uv_map[:,2,:,:] = self.uv_detail_model(self.pred_detail_latents)[:,0,:,:]*1e-2
                self.pred_uv_map = hsv_to_rgb(coarse_uv_map).cuda()
            # render front face
            vertex_uv_coord = self.facemodel.vtx_vt.unsqueeze(0).repeat(self.pred_coeffs.size()[0], 1, 1)
            render_feat = torch.cat([vertex_uv_coord, self.pred_shading], axis=2)
            self.render_face_mask, _, self.render_face = \
                self.renderer(self.pred_vertex, self.facemodel.face_buf, feat=render_feat, uv_map=self.pred_uv_map)
        else:
            # render front face
            self.render_face_mask, _, self.render_face = \
                self.renderer(self.pred_vertex, self.facemodel.face_buf, feat=self.pred_color)

    def visualize(self, input_data, is_uv_tex=True):
        # input data
        input_img = tensor2np(input_data['img'][:1, :, :, :])
        skin_img = img3channel(tensor2np(input_data['skin_mask'][:1, :, :, :]))
        parse_mask = tensor2np(input_data['parse_mask'][:1, :, :, :], dst_range=1.0)
        gt_lm = input_data['lm'][0, :, :].detach().cpu().numpy()
        # predict data
        pred_face_img = self.render_face * self.render_face_mask + (1 - self.render_face_mask) * input_data['img']
        pred_face_img = tensor2np(pred_face_img[:1, :, :, :])
        pred_lm = self.pred_lm[0, :, :].detach().cpu().numpy()
        # draw mask and landmarks
        parse_img = draw_mask(input_img, parse_mask)
        gt_lm[..., 1] = pred_face_img.shape[0] - 1 - gt_lm[..., 1]
        pred_lm[..., 1] = pred_face_img.shape[0] - 1 - pred_lm[..., 1]
        lm_img = draw_landmarks(pred_face_img, gt_lm, color='b')
        lm_img = draw_landmarks(lm_img, pred_lm, color='r')
        # combine visual images
        combine_img = np.concatenate([input_img, skin_img, parse_img, lm_img, pred_face_img], axis=1)
        if is_uv_tex:
            pre_uv_img = tensor2np(F.interpolate(self.pred_uv_map, size=input_img.shape[:2], mode='area')[:1, :, :, :])
            combine_img = np.concatenate([combine_img, pre_uv_img], axis=1)
        return combine_img

    def visualize_3dmmtex_as_uv(self):
        tex_vertex = self.pred_tex[0, :, :].detach().cpu().numpy()
        unwrap_uv_idx_v_idx = self.facemodel.unwrap_uv_idx_v_idx.detach().cpu().numpy()
        unwrap_uv_idx_bw = self.facemodel.unwrap_uv_idx_bw.detach().cpu().numpy()
        tex_uv = unwrap_vertex_to_uv(tex_vertex, unwrap_uv_idx_v_idx, unwrap_uv_idx_bw) * 255.
        return tex_uv

    def save_mesh(self, path, mesh_name, mlt_name=None, uv_name=None, is_uv_tex=True):
        pred_coeffs_dict = self.facemodel.split_coeff(self.pred_coeffs)
        pred_id_vertex, pred_exp_vertex, pred_alb_tex = self.facemodel.compute_for_mesh(pred_coeffs_dict)
        if is_uv_tex:
            assert mlt_name is not None and uv_name is not None
            write_mtl(os.path.join(path, mlt_name), uv_name)
            id_mesh_info = {
                'v': pred_id_vertex.detach()[0].cpu().numpy(),
                'vt': self.facemodel.vt_list.cpu().numpy(),
                'fv': self.facemodel.head_buf.cpu().numpy(),
                'fvt': self.facemodel.head_tri_vt.cpu().numpy(),
                'mtl_name': mlt_name
            }
            exp_mesh_info = {
                'v': pred_exp_vertex.detach()[0].cpu().numpy(),
                'vt': self.facemodel.vt_list.cpu().numpy(),
                'fv': self.facemodel.head_buf.cpu().numpy(),
                'fvt': self.facemodel.head_tri_vt.cpu().numpy(),
                'mtl_name': mlt_name
            }
        else:
            id_mesh_info = {
                'v': pred_id_vertex.detach()[0].cpu().numpy(),
                'vt': pred_alb_tex.detach()[0].cpu().numpy(),
                'fv': self.facemodel.head_buf.cpu().numpy()
            }
            exp_mesh_info = {
                'v': pred_exp_vertex.detach()[0].cpu().numpy(),
                'vt': pred_alb_tex.detach()[0].cpu().numpy(),
                'fv': self.facemodel.head_buf.cpu().numpy()
            }
        write_mesh_obj(mesh_info=id_mesh_info, file_path=os.path.join(path, f'{mesh_name[:-4]}_id{mesh_name[-4:]}'))
        write_mesh_obj(mesh_info=exp_mesh_info, file_path=os.path.join(path, f'{mesh_name[:-4]}_exp{mesh_name[-4:]}'))

    def save_coeffs(self, path, coeffs_name, is_uv_tex=True):
        # coeffs & landmarks
        coeffs_info = {'coeffs': self.pred_coeffs, 'lm68': self.pred_lm}
        if is_uv_tex:
            coeffs_info['latents_w'] = self.pred_latents_w
            coeffs_info['latents_z'] = self.pred_latents_z
        torch.save(coeffs_info, os.path.join(path, coeffs_name))

    def gather_args_str(self):
        args_str = '\n'
        for name in self.args_names:
            args_dict = getattr(self, 'args_' + name)
            args_str += f'----------------- Args-{name} ---------------\n'
            for k, v in args_dict.items():
                args_str += '{:>30}: {:<30}\n'.format(str(k), str(v))
        args_str += '----------------- End -------------------'
        return args_str

    def fitting(self, input_data, logger):
        if os.path.isfile(os.path.join(logger.vis_dir, 'stage2_coeffs.pt')):
            return
        # fix random seed
        setup_seed(123)

        # print args
        logger.write_txt_log(self.gather_args_str())

        # save the input data
        torch.save(input_data, os.path.join(logger.vis_dir, f'input_data.pt'))

        #--------- Stage 1 - getting initial coeffs by Deep3D NN inference ---------

        logger.write_txt_log('Stage 1 getting initial coeffs by Deep3D NN inference.')
        with torch.no_grad():
            self.pred_coeffs = self.net_recon_deep3d(input_data['img'].to(self.device))
            print(self.pred_coeffs)
        self.infer_render(is_uv_tex=False)
        vis_img = self.visualize(input_data, is_uv_tex=False)
        vis_tex_uv = self.visualize_3dmmtex_as_uv()
        logger.write_disk_images([vis_img], ['stage1_vis'])
        logger.write_disk_images([vis_tex_uv], ['stage1_vis_3dmmtex_as_uv'])
        self.save_mesh(path=logger.vis_dir, mesh_name='stage1_mesh.obj', is_uv_tex=False)
        self.save_coeffs(path=logger.vis_dir, coeffs_name='stage1_coeffs.pt', is_uv_tex=False)

        #--------- Stage 2 - search UV tex on a spherical surface with fixed shape ---------

        logger.write_txt_log('Start stage 2 searching UV tex on a spherical surface with fixed shape.')
        logger.reset_prefix(prefix='s2_search_uvtex_spherical_fixshape')
        fitter = uvtex_spherical_fixshape_fitter.Fitter(facemodel=self.facemodel,
                                                        uv3dmm_model=self.uv3dmm_model,
                                                        renderer=self.renderer,
                                                        net_recog=self.net_recog,
                                                        net_vgg=self.net_vgg,
                                                        logger=logger,
                                                        input_data=input_data,
                                                        init_coeffs=self.pred_coeffs,
                                                        init_latents_z=None,
                                                        **self.args_s2_search_uvtex_spherical_fixshape)
        self.pred_coeffs, self.pred_latents_z, self.pred_latents_w = fitter.iterate()
        logger.reset_prefix()
        logger.write_txt_log('End stage 2 searching UV tex on a spherical surface with fixed shape.')

        self.infer_render()
        vis_img = self.visualize(input_data)
        logger.write_disk_images([vis_img], ['stage2_vis'])
        logger.write_disk_images([tensor2np(self.pred_uv_map[:1, :, :, :])], ['stage2_uv'])
        self.save_mesh(path=logger.vis_dir,
                       mesh_name='stage2_mesh.obj',
                       mlt_name='stage2_mesh.mlt',
                       uv_name='stage2_uv.png')
        self.save_coeffs(path=logger.vis_dir, coeffs_name='stage2_coeffs.pt')

        #----------------------------------------新---------------------------
        # return
        texture_detail_fitting = True
        
        # self.uv_3dmm=
        if texture_detail_fitting:
            logger.write_txt_log('Start stage 3 args_s3_predict_detail_uvtex')
            logger.reset_prefix(prefix='args_s3_predict_detail_uvtex')
            fitter = uvtex_detail_fitter.Fitter(facemodel=self.facemodel,
                                                uv3dmm_model=self.uv3dmm_model,
                                                uv_detail_model=self.uv_detail_model,
                                                renderer=self.renderer,
                                                net_recog=self.net_recog,
                                                net_vgg=self.net_vgg,
                                                logger=logger,
                                                input_data=input_data,
                                                init_coeffs=self.pred_coeffs,
                                                init_latents_z=self.pred_latents_z,
                                                init_detail_coeffs=None,
                                                **self.args_s3_predict_detail_uvtex)
            self.pred_coeffs, self.pred_latents_z, self.pred_detail_latents = fitter.iterate()
            logger.reset_prefix()
            logger.write_txt_log('End stage 3 predict detail uvtex.')

            self.infer_render()
            vis_img = self.visualize(input_data)
            logger.write_disk_images([vis_img], ['stage3_vis'])
            logger.write_disk_images([tensor2np(self.pred_uv_map[:1, :, :, :])], ['stage3_uv'])
            self.save_mesh(path=logger.vis_dir,
                        mesh_name='stage3_mesh.obj',
                        mlt_name='stage3_mesh.mlt',
                        uv_name='stage3_uv.png')
            self.save_coeffs(path=logger.vis_dir, coeffs_name='stage3_coeffs.pt')
            self.pred_detail_latents=None

        #---------------------------------------------------------------------
        return
        #--------- Stage 3 - jointly optimize UV tex and shape ---------

        logger.write_txt_log('Start stage 3 jointly optimize UV tex and shape.')
        logger.reset_prefix(prefix='s3_optimize_uvtex_shape_joint')
        fitter = uvtex_wspace_shape_joint_fitter.Fitter(facemodel=self.facemodel,
                                                        tex_gan=self.tex_gan,
                                                        renderer=self.renderer,
                                                        net_recog=self.net_recog,
                                                        net_vgg=self.net_vgg,
                                                        logger=logger,
                                                        input_data=input_data,
                                                        init_coeffs=self.pred_coeffs,
                                                        init_latents_z=self.pred_latents_z,
                                                        **self.args_s3_optimize_uvtex_shape_joint)
        self.pred_coeffs, self.pred_latents_z, self.pred_latents_w = fitter.iterate()
        logger.reset_prefix()
        logger.write_txt_log('End stage 3 jointly optimize UV tex and shape.')

        self.infer_render()
        vis_img = self.visualize(input_data)
        logger.write_disk_images([vis_img], ['stage3_vis'])
        logger.write_disk_images([tensor2np(self.pred_uv_map[:1, :, :, :])], ['stage3_uv'])
        self.save_mesh(path=logger.vis_dir,
                       mesh_name='stage3_mesh.obj',
                       mlt_name='stage3_mesh.mlt',
                       uv_name='stage3_uv.png')
        self.save_coeffs(path=logger.vis_dir, coeffs_name='stage3_coeffs.pt')
