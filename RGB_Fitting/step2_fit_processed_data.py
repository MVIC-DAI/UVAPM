import os
import time
import argparse
import torch

from model import ours_fit_model

from utils.visual_utils import Logger
from utils.data_utils import setup_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir',
                        type=str,
                        default='../data/fitting_examples/inputs/processed_data',
                        help='directory of input data')
    parser.add_argument('--output_dir',
                        type=str,
                        default='../data/fitting_examples/outputs',
                        help='directory of outputs')
    parser.add_argument('--checkpoints_dir', type=str, default='../checkpoints', help='pretrained models.')
    parser.add_argument('--topo_dir', type=str, default='../topo_assets', help='assets of topo.')
    parser.add_argument('--uv3dmm_model_dir', type=str, default='./model/uv3dmm', help='uv3dmm model directory.')
    parser.add_argument('--device', type=str, default='cuda', help='cuda/cpu')
    args = parser.parse_args()

    setup_seed(123)  # fix random seed

    fit_model = ours_fit_model.FitModel(cpk_dir=args.checkpoints_dir,
                                        topo_dir=args.topo_dir,
                                        uv3dmm_model_dir=args.uv3dmm_model_dir,
                                        device=args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    fnames = [fn for fn in sorted(os.listdir(args.input_dir)) if fn.endswith('.pt')]

    for fn in fnames:
        basename = fn[:fn.rfind('.')]

        os.makedirs(os.path.join(args.output_dir, basename), exist_ok=True)

        logger = Logger(
            vis_dir=os.path.join(args.output_dir, basename),
            flag=f'uv_3dmm_fitting',
            is_tb=True)
        logger.write_txt_log(f'Fit image: {os.path.join(args.input_dir, fn)}')

        input_data = torch.load(os.path.join(args.input_dir, fn))
        if 'trans_params' in input_data:
            input_data.pop('trans_params')
        input_data = {k: v.to(args.device) for (k, v) in input_data.items()}
        tic = time.time()
        fit_model.fitting(input_data=input_data, logger=logger)
        toc = time.time()

        logger.write_txt_log(f'Fit image: {fn} done, took {toc - tic:.4f} seconds.')
        logger.close()
