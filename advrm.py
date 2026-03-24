from images import *
from load_model import load_target_model, integrated_mde_models, predict_batch, predict_depth_fn
import pandas as pd
import torchvision.models as models
from model import get_style_model_and_losses
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from log import log_img_train,log_scale_train,log_scale_eval, log_img_eval
import urllib
import traceback
from torch.optim import Adam
import torch.optim as optim
from lr_decay import PolynomialLRDecay
from skimage.metrics import structural_similarity as ssim_metric
import torchvision.utils as vutils
import os
import torch.nn.functional as F

def bim(grad, input_patch, lr):  
    input_patch=(torch.clamp(input_patch-lr*grad.sign(),0,1)).detach() 
    input_patch.requires_grad_()
    return input_patch

def mifgsm(grad, mo, input_patch, lr):
    raise NotImplementedError

class ADVRM:
    def __init__(self,args, log, patch_size=None, rewrite=False, random_object_flag=True) -> None:
        self.args = args
        self.log = log
        self.patch = Patch(self.args)
        self.patch.load_patch_warp(self.args['patch_file'], self.args['patch_dir'], patch_size, rewrite = rewrite)
        self.objects = OBJ(self.args)
        if random_object_flag:
            self.objects.load_obj_warp(['pas','car','obs'])
            print('==================objects for training============')
            print(self.objects.object_files_train)
            print('==================objects for test============')
            print(self.objects.object_files_test)
            self.run = self.run_with_random_object
        else:
            self.run = self.run_with_fixed_object
        self.MDE = load_target_model(self.args['depth_model'], self.args)
        self.configure_loss(self.patch)

    def configure_loss(self, patch):
        def adv_loss_fn(batch, batch_y, target_depth):
            _, _, _, _, object_full_mask = batch
            adv_depth = batch_y[0] 
            
            B = object_full_mask.shape[0]
            merging_loss = 0.0
            
            first_mask_up = None
            first_mask_bt = None
            
            for i in range(B):
                mask = object_full_mask[i, 0]  # (H, W)
                
                bg_depth = target_depth[i, 0] if target_depth.dim() == 4 else target_depth[i].squeeze()
                cur_adv_depth = adv_depth[i, 0] if adv_depth.dim() == 4 else adv_depth[i].squeeze()
                
                if mask.sum() == 0:
                    continue
                    
                # 차량 영역의 순수 배경 평균 깊이 계산
                mean_bg_depth = (bg_depth * mask).sum() / (mask.sum() + 1e-7)
                
                H, W = mask.shape
                # 이미지 전체의 Y좌표, X좌표를 담은 행렬 생성
                y_coords, x_coords = torch.meshgrid(torch.arange(H, device=mask.device), torch.arange(W, device=mask.device), indexing='ij')
                
                # 좌/우 윤곽선에서 Reference Point 2개 찾기
            
                # 마스크가 없는 곳은 무한대/음수로 채워서 좌/우 윤곽선 X좌표 추출
                x_masked_left = x_coords.float().masked_fill(mask == 0, float('inf'))
                x_left_contour = x_masked_left.min(dim=1).values # 각 Y 높이별 가장 왼쪽 X좌표 (H,)
                
                x_masked_right = x_coords.float().masked_fill(mask == 0, -1.0)
                x_right_contour = x_masked_right.max(dim=1).values # 각 Y 높이별 가장 오른쪽 X좌표 (H,)
                
                # 마스크가 존재하는 유효한 Y좌표(높이)들만 추출
                valid_y = torch.nonzero(x_left_contour != float('inf')).squeeze(-1)
                
                if len(valid_y) < 2:
                    continue
                    
                # 유효한 윤곽선 상의 픽셀 X좌표와 해당 위치의 배경 깊이 가져오기
                left_x = x_left_contour[valid_y].long()
                right_x = x_right_contour[valid_y].long()
                
                left_depths = bg_depth[valid_y, left_x]
                right_depths = bg_depth[valid_y, right_x]
                
                # 평균 배경 깊이와 가장 비슷한 좌/우 각각의 인덱스 찾기
                left_idx = torch.argmin(torch.abs(left_depths - mean_bg_depth))
                right_idx = torch.argmin(torch.abs(right_depths - mean_bg_depth))
                
                # 찾은 두 점 (X_L, Y_L)과 (X_R, Y_R)
                y_L, x_L = valid_y[left_idx].float(), left_x[left_idx].float()
                y_R, x_R = valid_y[right_idx].float(), right_x[right_idx].float()
                
                # 두 점을 잇는 직선의 기울기(m) 계산
                if x_R == x_L:
                    m = 0.0
                else:
                    m = (y_R - y_L) / (x_R - x_L)
                    
                # 이미지 전체 픽셀 공간에 대한 '기준 사선(Y_line)' 행렬 계산
                Y_line = m * (x_coords.float() - x_L) + y_L
                
                # Y좌표가 기준 사선보다 작거나 같으면(위쪽이면) mask_up, 크면(아래쪽이면) mask_bt
                mask_up = ((mask > 0) & (y_coords.float() <= Y_line)).float()
                mask_bt = ((mask > 0) & (y_coords.float() > Y_line)).float()
                
                if i == 0:
                    first_mask_up = mask_up.clone()
                    first_mask_bt = mask_bt.clone()
                
                loss_up = F.mse_loss(cur_adv_depth * mask_up, bg_depth * mask_up, reduction='sum') / (mask_up.sum() + 1e-7)
                loss_bt = F.mse_loss(cur_adv_depth * mask_bt, bg_depth * mask_bt, reduction='sum') / (mask_bt.sum() + 1e-7)
                
                merging_loss += (loss_up + loss_bt)
            
            if first_mask_up is not None:
                self.current_mask_up = first_mask_up.unsqueeze(0)
                self.current_mask_bt = first_mask_bt.unsqueeze(0)
                
            return merging_loss / B

        def patch_style_core(style_image, content_image, style_mask, content_mask, laplacian_m):
            cnn = models.vgg19(weights='DEFAULT').features.cuda(self.args['device']).eval()
            cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).cuda(self.args['device'])
            cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).cuda(self.args['device'])
            style_model, style_losses, content_losses, tv_losses = get_style_model_and_losses(cnn,
                cnn_normalization_mean, cnn_normalization_std, style_image, content_image, style_mask, content_mask, laplacian_m)

            for param in style_model.parameters():
                param.requires_grad = False
            return style_model, style_losses, content_losses, tv_losses
            
        def total_variation_loss(img):
            bs_img, c_img, h_img, w_img = img.size()
            tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
            tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
            return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img)
            
        self.style_model, self.style_losses, self.content_losses, _  = patch_style_core(patch.patch_img_s, patch.patch_img_c,torch.torch.ones([1,patch.patch_img_s.shape[2],patch.patch_img_s.shape[3]]).cuda(self.args['device']), torch.torch.ones([1,patch.patch_img_s.shape[2],patch.patch_img_s.shape[3]]).cuda(self.args['device']), patch.laplacian_m)
        self.tv_loss_fn = total_variation_loss
        self.adv_loss_fn = adv_loss_fn
       
    def compute_sty_loss(self, patch, gt, style_weight, content_weight, tv_weight):
        style_score = torch.zeros(1).float().cuda(self.args['device'])
        content_score = torch.zeros(1).float().cuda(self.args['device'])
        tv_score = torch.zeros(1).float().cuda(self.args['device']) 
        self.style_model(patch)
        for sl in self.style_losses:
            style_score += sl.loss
        for cl in self.content_losses:
            content_score += cl.loss
        tv_score+=self.tv_loss_fn(patch-gt)
        style_loss = style_weight * style_score + content_weight * content_score + tv_weight * tv_score 
        return style_loss, style_score, content_score, tv_score

    def run_with_random_object(self, scene_dir, scene_file, idx, points):
        self.env = ENV(self.args, scene_file, scene_dir, idx, points)
        name_prefix = f"{self.args['patch_file'][:-4]}_{scene_file[:-4]}_{idx}"
        if self.args['update']=='lbfgs':
            optimizer = optim.LBFGS([self.patch.optmized_patch], lr=self.args['learning_rate'])
            LR_decay = PolynomialLRDecay(optimizer, self.args['epoch']//2, self.args['learning_rate']/2, 0.9)

        final_mrsr = 0.0
        final_ssim = 0.0

        clean_env = F.interpolate(self.env.env, size=([int(self.args['input_height']), int(self.args['input_width'])]))
        dummy_mask = torch.zeros_like(clean_env[:, 0:1, :, :])
        with torch.no_grad():
            bg_depth_res = predict_batch([clean_env, clean_env, clean_env, dummy_mask, dummy_mask], self.MDE)
            pure_bg_depth = bg_depth_res[0].detach()

        for epoch in tqdm(range(self.args['epoch']), desc=f"Training {idx}/{self.args['scene_num']}"):
            def closure(): 
                nonlocal final_mrsr, final_ssim

                self.patch.optmized_patch.data.clamp_(0, 1)
                batch, patch_size = self.env.accept_patch_and_objects(False, self.patch.optmized_patch, self.patch.mask, self.objects.object_imgs_train, self.env.insert_range, None, None, offset_patch=self.args['train_offset_patch_flag'], color_patch=self.args['train_color_patch_flag'], offset_object=self.args['train_offset_object_flag'], color_object=self.args['train_color_object_flag'])

                adv_scene_image, ben_scene_image, scene_img, patch_full_mask, object_full_mask = batch
                batch_resized = [ F.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch]
                batch_y = predict_batch(batch_resized, self.MDE)

                with torch.no_grad():
                    target_depth = pure_bg_depth 
                    ben_diff = (batch_y[1] - target_depth) * object_full_mask
                    ben_bg_rmse = torch.sqrt(torch.sum(ben_diff**2) / (torch.sum(object_full_mask) + 1e-7))
                    adv_diff = (batch_y[0] - target_depth) * object_full_mask
                    adv_bg_rmse = torch.sqrt(torch.sum(adv_diff**2) / (torch.sum(object_full_mask) + 1e-7))

                if epoch % 50 == 0 or (epoch+1) == self.args['epoch']:
                    tqdm.write(f"[Epoch {epoch:3d}] 원본 갭: {ben_bg_rmse.item():.4f} | 공격 후 갭: {adv_bg_rmse.item():.4f} ➔ 0 수렴 목표")
                    
                if self.args['train_log_flag']:
                    self.log.add_scalar(f'{name_prefix}/debug/benign_bg_gap_rmse', ben_bg_rmse.item(), epoch)
                    self.log.add_scalar(f'{name_prefix}/debug/adv_bg_gap_rmse', adv_bg_rmse.item(), epoch)
            
                _, _, _, e_blend, e_cover = self.eval_core(batch_y[0], batch_y[1], batch_resized[-1])
                
                adv_loss = self.adv_loss_fn(batch_resized, batch_y, target_depth)
                
                style_loss, style_score, content_score, tv_score = self.compute_sty_loss(self.patch.optmized_patch,  self.patch.init_patch, self.args['style_weight'], self.args['content_weight'], self.args['tv_weight'])
                
                loss = self.args['lambda']*style_loss + adv_loss
                if self.args['update']=='lbfgs':
                    loss.backward()

                if self.args['train_quan_patch_flag'] and (epoch+1)%10==0:
                    optmized_patch = (self.patch.optmized_patch*255).int().float()/255.
                    self.patch.optmized_patch.data = optmized_patch.data
                
                if self.args['train_log_flag']:
                    if epoch % self.args['train_img_log_interval']==0 or (epoch+1)==self.args['epoch']:
                        log_img_train(self.log, epoch, name_prefix, [adv_scene_image, ben_scene_image, (self.patch.optmized_patch*255).int().float()/255., batch_y[0], batch_y[1], clean_env, target_depth])
                        self.log.add_image(f'{name_prefix}/train/mask_1_full', object_full_mask.detach().cpu()[0, 0], epoch, dataformats='HW')
                        
                        if hasattr(self, 'current_mask_up') and hasattr(self, 'current_mask_bt'):
                            self.log.add_image(f'{name_prefix}/train/mask_2_upper', self.current_mask_up.detach().cpu()[0], epoch, dataformats='HW')
                            self.log.add_image(f'{name_prefix}/train/mask_3_bottom', self.current_mask_bt.detach().cpu()[0], epoch, dataformats='HW')
                        # log_scale_train(self.log,epoch, name_prefix, style_score, content_score, tv_score, adv_loss, mean_shift, max_shift, min_shift, mean_ori, arr)
                        log_scale_train(self.log,epoch, name_prefix, style_score, content_score, tv_score, adv_loss, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), e_blend, e_cover)
                
                if self.args['inner_eval_flag']:
                    if epoch % self.args['inner_eval_interval']==0 or (epoch+1)==self.args['epoch']: 
                        for category in self.objects.object_imgs_test.keys():
                            if self.args['random_test_flag']:
                                record = [[] for i in range(5)]
                                for _ in range(20):
                                    record_tmp = self.eval(self.MDE, category)
                                    for i in range(5):
                                        record[i]+=record_tmp[i]       
                            else:
                                record = self.eval(self.MDE, category)
                            
                            log_scale_eval(self.log, epoch, name_prefix, self.MDE[0],category, np.mean(record[0]), np.mean(record[1]), np.mean(record[2]),np.mean(record[3]),np.mean(record[4]) )      

                            final_mrsr = np.mean(record[3])

                            if (epoch + 1) == self.args['epoch']:
                                patch_opt = self.patch.optmized_patch.detach().cpu().squeeze().permute(1, 2, 0).numpy()
                                patch_init = self.patch.init_patch.detach().cpu().squeeze().permute(1, 2, 0).numpy()
                                final_ssim = ssim_metric(patch_init, patch_opt, data_range=1.0, channel_axis=-1)

                                save_dir = "./saved_patches"
                                os.makedirs(save_dir, exist_ok=True) 
                                trial_name = self.args.get('log_dir_comment', f'default_patch').strip()
                                save_path = os.path.join(save_dir, f"{trial_name}_scene_{idx}.png")
                                vutils.save_image(self.patch.optmized_patch.data, save_path)

                return loss

            if self.args['update']=='lbfgs':
                optimizer.zero_grad()
                optimizer.step(closure)
                LR_decay.step()
            else:
                loss = closure()
                grad = torch.autograd.grad(loss, [self.patch.optmized_patch] )[0]
                self.patch.optmized_patch = bim(grad, self.patch.optmized_patch,self.args['learning_rate'])

        return final_mrsr, final_ssim

    def run_with_fixed_object(self, scene_dir, scene_file, idx, points):
        self.env = ENV(self.args, scene_file, scene_dir, idx, points)
        name_prefix = f"{self.args['patch_file'][:-4]}_{scene_file[:-4]}_{idx}"
        if self.args['update']=='lbfgs':
            optimizer = optim.LBFGS([self.patch.optmized_patch], lr=self.args['learning_rate'])
            LR_decay = PolynomialLRDecay(optimizer, self.args['epoch']//2, self.args['learning_rate']/2, 0.9)

        clean_env = F.interpolate(self.env.env, size=([int(self.args['input_height']), int(self.args['input_width'])]))
        dummy_mask = torch.zeros_like(clean_env[:, 0:1, :, :])
        with torch.no_grad():
            bg_depth_res = predict_batch([clean_env, clean_env, clean_env, dummy_mask, dummy_mask], self.MDE)
            pure_bg_depth = bg_depth_res[0].detach()

        for epoch in tqdm(range(self.args['epoch']), desc=f"Training {idx}/{self.args['scene_num']}"):
            def closure(): 
                if self.args['update']=='lbfgs':
                    optimizer.zero_grad()
                self.patch.optmized_patch.data.clamp_(0, 1)
                adv_scene_image, ben_scene_image, patch_size, patch_full_mask = self.env.accept_patch(self.patch.optmized_patch, None, self.patch.mask, self.args['insert_height'], offset_patch=self.args['train_offset_patch_flag'], color_patch=self.args['train_color_patch_flag'])

                batch= [adv_scene_image, ben_scene_image, self.env.env, patch_full_mask, self.objects.object_full_mask]
                batch_resized = [ F.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch]

                batch_y = predict_batch(batch_resized, self.MDE)
                
                with torch.no_grad():
                    target_depth = pure_bg_depth

                # mean_ori, mean_shift, max_shift, min_shift, arr =self.eval_core(batch_y[0], batch_y[1], batch_resized[-1])
                _, _, _, e_blend, e_cover = self.eval_core(batch_y[0], batch_y[1], batch_resized[-1])
                
                adv_loss = self.adv_loss_fn(batch_resized, batch_y, target_depth)
                
                style_loss, style_score, content_score, tv_score = self.compute_sty_loss(self.patch.optmized_patch,  self.patch.init_patch, self.args['style_weight'], self.args['content_weight'], self.args['tv_weight'])
                
                loss = self.args['lambda']*style_loss + adv_loss
                if self.args['update']=='lbfgs':
                    loss.backward()

                if self.args['train_quan_patch_flag'] and (epoch+1)%10==0:
                    optmized_patch = (self.patch.optmized_patch*255).int().float()/255.
                    self.patch.optmized_patch.data = optmized_patch.data
                if self.args['train_log_flag']:
                    if epoch % self.args['train_img_log_interval']==0 or (epoch+1)==self.args['epoch']:
                        log_img_train(self.log, epoch, name_prefix, [adv_scene_image, ben_scene_image, (self.patch.optmized_patch*255).int().float()/255., batch_y[0], batch_y[1], clean_env, target_depth])
                        log_scale_train(self.log,epoch, name_prefix, style_score, content_score, tv_score, adv_loss, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), e_blend, e_cover)
                
                return loss
            if self.args['update']=='lbfgs':
                optimizer.step(closure)
                LR_decay.step()
            else:
                loss = closure()
                grad = torch.autograd.grad(loss, [self.patch.optmized_patch] )[0]
                self.patch.optmized_patch = bim(grad, self.patch.optmized_patch,self.args['learning_rate'])
    
    def eval(self, MDE, category, insert_height=None, patch=None):
     if patch is None:
         patch = self.patch.optmized_patch
     if self.args['test_quan_patch_flag']:
         patch = (patch*255).int().float()/255.

     category = 'car'
     object_num = 1

     with torch.no_grad():
         record = [[] for _ in range(5)]

         for idx in range(len(self.objects.object_imgs_test[category]) - object_num + 1):
             batch, _ = self.env.accept_patch_and_objects(
                 True, patch, self.patch.mask, 
                 self.objects.object_imgs_test,
                 self.env.insert_range, insert_height, None, 
                 offset_patch=False, color_patch=False, offset_object=False, color_object=False,
                 object_idx_g=idx, 
                 category=category
             )

             batch= [ F.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch]
             batch_y= predict_batch(batch, MDE)

             if idx == 0 and self.args['train_log_flag']:
                 name_prefix = f"eval_{self.args['patch_file'][:-4]}_{category}"
                 log_img_eval(self.log, self.args['epoch'], name_prefix, [batch[0], batch[1], (patch*255).int().float()/255., batch_y[0], batch_y[1], batch[2], batch_y[1]])

             _, _, _, e_blend, e_cover = self.eval_core(batch_y[0], batch_y[1], batch[-1])
         
             record[0].append(0.0)
             record[1].append(0.0)
             record[2].append(0.0)
             record[3].append(e_blend.item())
             record[4].append(e_cover.item()) 
         

         return record

    def eval_core(self, adv_depth, ref_depth, scene_obj_mask):
        if adv_depth.dim() == 3:
            B, H, W = adv_depth.shape
        else:
            B, C, H, W = adv_depth.shape
            
        tau = 20.0
        
        e_blend_batch = 0.0
        e_cover_batch = 0.0
        
        for b in range(B):
            d_adv = adv_depth[b, 0] if adv_depth.dim() == 4 else adv_depth[b]
            d_ben = ref_depth[b, 0] if ref_depth.dim() == 4 else ref_depth[b]
            m = scene_obj_mask[b, 0] if scene_obj_mask.dim() == 4 else scene_obj_mask[b]
            
            if m.sum() == 0:
                continue
            
            # 1. E_cover 계산 (Perturbation Coverage Ratio)
            cover_mask = (d_adv != d_ben) & (m > 0)
            e_cover = cover_mask.sum().float() / (m.sum() + 1e-7)
            e_cover_batch += e_cover
            
            # 2. E_blend 계산 (Hiding Error)
            y_coords, x_coords = torch.meshgrid(torch.arange(H, device=m.device), torch.arange(W, device=m.device), indexing='ij')
            
            x_masked_left = x_coords.float().masked_fill(m == 0, float('inf'))
            x_left_contour = x_masked_left.min(dim=1).values
            
            x_masked_right = x_coords.float().masked_fill(m == 0, -1.0)
            x_right_contour = x_masked_right.max(dim=1).values
            
            valid_y = torch.nonzero(x_left_contour != float('inf')).squeeze(-1)
            
            if len(valid_y) == 0:
                continue
                
            x_left_bg = torch.clamp(x_left_contour[valid_y] - 1, min=0).long()
            x_right_bg = torch.clamp(x_right_contour[valid_y] + 1, max=W-1).long()
            
            d_bg_left = d_ben[valid_y, x_left_bg]
            d_bg_right = d_ben[valid_y, x_right_bg]
            
            # 행(row)별 윤곽선 평균 깊이 계산
            d_bg_bar = (d_bg_left + d_bg_right) / 2.0
            
            d_bg_map = torch.zeros_like(d_adv)
            d_bg_map[valid_y, :] = d_bg_bar.unsqueeze(1)
            
            # 임계값 tau보다 크게 차이나는 픽셀 비율 계산
            blend_mask = (torch.abs(d_adv - d_bg_map) > tau) & (m > 0)
            e_blend = blend_mask.sum().float() / (m.sum() + 1e-7)
            e_blend_batch += e_blend
            
        e_blend_mean = torch.tensor(e_blend_batch / B).to(adv_depth.device)
        e_cover_mean = torch.tensor(e_cover_batch / B).to(adv_depth.device)
        
        return torch.tensor(0.0).to(adv_depth.device), torch.tensor(0.0).to(adv_depth.device), torch.tensor(0.0).to(adv_depth.device), e_blend_mean, e_cover_mean