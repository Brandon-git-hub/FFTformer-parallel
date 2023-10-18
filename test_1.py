# deepspeed --num_gpu 2 test_1.py
import os
import torch
import argparse
from basicsr.models.archs.fftformer_arch import  fftformer, fftformerpipe, fftformerSequential
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image as Image
from tqdm import tqdm
from torch import autocast
from skimage.metrics import peak_signal_noise_ratio

import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed import comm as dist

from typing import cast
# from torchgpipe import GPipe
# from torchgpipe.balance import balance_by_time


# deepspeed.init_distributed(dist_backend='nccl')
# dist.barrier()
# world_size = dist.get_world_size()

def CenterCrop(image, resolution):
    #image = Image.open('1.png').convert("RGB")
    
    # resolution = 512
    w, h = image.size
    left = (w-resolution)/2
    top = (h-resolution)/2
    right = left + resolution
    bottom = top + resolution
    im = image.crop((left, top, right, bottom))
    #im.save("test.png","png")

    return im

def main(args):
    # CUDNN
    # cudnn.benchmark = True


    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = fftformer()
    # model = fftformerSequential().to_layers().eval()
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict,strict = True)
    # model.cuda()

    # ========================================
    # model = cast(torch.nn.Sequential, model)   
    # print('AUTO BALANCE')
    # partitions = torch.cuda.device_count()
    # print('Number of GPUS: ', partitions)

    # sample = torch.rand(1,3,128,128)
    # BALANCE = balance_by_time(partitions, model, sample)
    # print('Balance: ', BALANCE) # Balance:  [10, 11]
    # BALANCE = [13, 8]
    
    #@pytest.mark.parametrize('checkpoint', ['never', 'always', 'except_last'])
    # model = GPipe(model, balance=BALANCE, chunks=1, checkpoint='never')

    # # In and Out devices
    # in_device = model.devices[0]
    # out_device = model.devices[-1]
    # print('in_device: ', in_device)
    # print('out_device: ', out_device)


    # ===========================================
    # for k,v in state_dict.items():
    #     print(k,v.dtype)
    
    
    
    # net = PipelineModule(layers=model, num_stages=1)
    # # , replace_with_kernel_inject=True,partition_method='parameters'   
    # engine = deepspeed.init_inference(net, mp_size=2, dtype=torch.half, replace_with_kernel_inject=True)


    _eval(model, args)

def _eval(model, args):

    image = Image.open('/media/user/VCLAB/dataset/GOPRO/test/blur/GOPR0384_11_00_000001.png')
    # image = CenterCrop(image, 512)
    input_img = F.to_tensor(image)
    input_img = torch.unsqueeze(input_img, 0)
    # input_img = input_img.half()
    

    label = Image.open('/media/user/VCLAB/dataset/GOPRO/test/sharp/GOPR0384_11_00_000001.png')
    # label = CenterCrop(label, 512)
    label_img = F.to_tensor(label)
    # label_img = label_img.half()
    #torch.unsqueeze(label_img, 0)   
    
    device = torch.device('cuda')

    torch.cuda.empty_cache()    
    
    
    precision_scope = autocast 
    with torch.no_grad():
        with precision_scope("cuda"):
            # Main Evaluation
            name = '0.png'
            input_img = input_img.to(device)

            b, c, h, w = input_img.shape
            h_n = (32 - h % 32) % 32
            w_n = (32 - w % 32) % 32
            input_img = torch.nn.functional.pad(input_img, (0, w_n, 0, h_n), mode='reflect')


            pred = model(input_img)
            torch.cuda.synchronize()
            pred = pred[:, :, :h, :w]

            pred_clip = torch.clamp(pred, 0, 1)

            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)
            print(f'PSNR:{psnr}')

            if args.save_image:
                save_name = os.path.join(args.result_dir, name)
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='fftformer', type=str)
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--data_dir', type=str, default='/media/user/VCLAB/dataset/GOPRO')

    # Test
    parser.add_argument('--test_model', type=str, default='./pretrain_model/fftformer_GoPro.pth')
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])

    args = parser.parse_args()
    args.result_dir = os.path.join('results/', args.model_name, 'GoPro/')
    print(args)
    main(args)
