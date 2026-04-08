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
import random

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
        def adv_loss_fn(d_adv, d_bg, m):
            # 세 텐서의 차원을 모두 [B, 1, H, W] 4차원으로 안전하게 강제 통일
            if d_adv.dim() == 3: d_adv = d_adv.unsqueeze(1)
            if d_bg.dim() == 3: d_bg = d_bg.unsqueeze(1)
            if m.dim() == 3: m = m.unsqueeze(1)
            
            # 마스크 영역을 1.0, 밖을 0.0으로 변환
            mask_float = (m > 0).float()
            
            # 오차 계산 후 마스크 곱하기 (마스크 밖의 픽셀은 오차가 0이 됨)
            diff = (d_adv - d_bg) * mask_float
            
            # 마스크 안쪽 픽셀들의 제곱 오차 평균 반환
            return torch.sqrt(torch.sum(diff ** 2) / (torch.sum(mask_float) + 1e-7))

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
        elif self.args['update']=='adam':
            optimizer = optim.Adam([self.patch.optmized_patch], lr=self.args['learning_rate'])

        # final_mrsr = 0.0
        final_e_blend = 0.0
        final_e_cover = 0.0
        final_ssim = 0.0

        clean_env = F.interpolate(self.env.env, size=([int(self.args['input_height']), int(self.args['input_width'])]))
        dummy_mask = torch.zeros_like(clean_env[:, 0:1, :, :])
        with torch.no_grad():
            bg_depth_res = predict_batch([clean_env, clean_env, clean_env, dummy_mask, dummy_mask], self.MDE)
            pure_bg_depth = bg_depth_res[0].detach()

        for epoch in tqdm(range(self.args['epoch']), desc=f"Training {idx}/{self.args['scene_num']}"):
            def closure(): 
                # nonlocal final_mrsr, final_ssim
                nonlocal final_e_blend, final_e_cover, final_ssim

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
                    tqdm.write(f"[Epoch {epoch:3d}] (ben-background) depth 차이 RMSE: {ben_bg_rmse.item():.4f} |  (adv-background) depth 차이 RMSE: {adv_bg_rmse.item():.4f} ➔ 0 수렴 목표")
                    
                if self.args['train_log_flag']:
                    self.log.add_scalar(f'{name_prefix}/debug/benign_bg_gap_rmse', ben_bg_rmse.item(), epoch)
                    self.log.add_scalar(f'{name_prefix}/debug/adv_bg_gap_rmse', adv_bg_rmse.item(), epoch)
        
                e_blend, e_cover = self.eval_core(batch_y[0], batch_y[1], target_depth, batch_resized[-1])
                
                adv_loss = self.adv_loss_fn(batch_y[0], target_depth, object_full_mask)
                
                style_loss, style_score, content_score, tv_score = self.compute_sty_loss(self.patch.optmized_patch,  self.patch.init_patch, self.args['style_weight'], self.args['content_weight'], self.args['tv_weight'])
                
                loss = self.args['lambda']*style_loss + adv_loss

                if self.args['update'] in ['lbfgs', 'adam']: 
                    loss.backward()

                if self.args['train_quan_patch_flag'] and (epoch+1)%10==0:
                    optmized_patch = (self.patch.optmized_patch*255).int().float()/255.
                    self.patch.optmized_patch.data = optmized_patch.data
                
                if self.args['train_log_flag']:
                    if epoch % self.args['train_img_log_interval']==0 or (epoch+1)==self.args['epoch']:
                    # if (epoch+1)==self.args['epoch']:
                        # log_img_train(self.log, epoch, name_prefix, [adv_scene_image, ben_scene_image, (self.patch.optmized_patch*255).int().float()/255., batch_y[0], batch_y[1], clean_env, target_depth])
                        log_imgs = [
                            adv_scene_image.detach().cpu(), 
                            ben_scene_image.detach().cpu(), 
                            ((self.patch.optmized_patch*255).int().float()/255.).detach().cpu(), 
                            batch_y[0].detach().cpu(), 
                            batch_y[1].detach().cpu(), 
                            clean_env.detach().cpu(), 
                            target_depth.detach().cpu()
                        ]
                        log_img_train(self.log, epoch, name_prefix, log_imgs)

                        self.log.add_image(f'{name_prefix}/train/mask_1_full', object_full_mask.detach().cpu()[0, 0], epoch, dataformats='HW')
                        
                        if hasattr(self, 'current_mask_up') and hasattr(self, 'current_mask_bt'):
                            self.log.add_image(f'{name_prefix}/train/mask_2_upper', self.current_mask_up.detach().cpu()[0], epoch, dataformats='HW')
                            self.log.add_image(f'{name_prefix}/train/mask_3_bottom', self.current_mask_bt.detach().cpu()[0], epoch, dataformats='HW')
                        # log_scale_train(self.log, epoch, name_prefix, style_score.item(), content_score.item(), tv_score.item(), adv_loss.item(), e_blend.item(), e_cover.item())
                        log_scale_train(self.log, epoch, name_prefix, style_score, content_score, tv_score, adv_loss, e_blend, e_cover)

                
                if self.args['inner_eval_flag']:
                    # if epoch % self.args['inner_eval_interval']==0 or (epoch+1)==self.args['epoch']: 
                    if (epoch+1)==self.args['epoch']: 
                        patch_opt = self.patch.optmized_patch.detach().cpu().squeeze().permute(1, 2, 0).numpy()
                        patch_init = self.patch.init_patch.detach().cpu().squeeze().permute(1, 2, 0).numpy()
                        final_ssim = ssim_metric(patch_init, patch_opt, data_range=1.0, channel_axis=-1)
                        self.args['current_ssim'] = final_ssim  # CSV 저장할 때 사용

                        # # 패치 이미지를 저장하는 로직도 위로 같이 끌어올려줘
                        # save_dir = "./saved_patches"
                        # os.makedirs(save_dir, exist_ok=True) 
                        # trial_name = self.args.get('log_dir_comment', f'default_patch').strip()
                        # save_path = os.path.join(save_dir, f"{trial_name}_scene_{idx}.png")
                        # vutils.save_image(self.patch.optmized_patch.data, save_path)

                        experiment_root_dir = os.path.dirname(self.log.log_dir)
                        experiment_name = os.path.basename(experiment_root_dir) # 예: 2026-04-07-18-57-51_multi_scene

                        save_dir = os.path.join("./saved_patches", experiment_name)
                        os.makedirs(save_dir, exist_ok=True) 

                        trial_name = self.args.get('log_dir_comment', f'default_patch').strip()
                        save_path = os.path.join(save_dir, f"{trial_name}_scene_{idx:03d}.png")
                        vutils.save_image(self.patch.optmized_patch.data, save_path)


                        for category in self.objects.object_imgs_test.keys():
                            if self.args['random_test_flag']:
                                record = [[] for i in range(2)]
                                for _ in range(20):
                                    record_tmp = self.eval(self.MDE, category)
                                    for i in range(2):
                                        record[i]+=record_tmp[i]       
                            else:
                                record = self.eval(self.MDE, category)
                            
                            log_scale_eval(self.log, epoch, name_prefix, self.MDE[0], category, np.mean(record[0]), np.mean(record[1]))

                            # final_mrsr = np.mean(record[0])
                            final_e_blend = np.mean(record[0])
                            final_e_cover = np.mean(record[1])

                            # if (epoch + 1) == self.args['epoch']:
                            #     patch_opt = self.patch.optmized_patch.detach().cpu().squeeze().permute(1, 2, 0).numpy()
                            #     patch_init = self.patch.init_patch.detach().cpu().squeeze().permute(1, 2, 0).numpy()
                            #     final_ssim = ssim_metric(patch_init, patch_opt, data_range=1.0, channel_axis=-1)

                            #     save_dir = "./saved_patches"
                            #     os.makedirs(save_dir, exist_ok=True) 
                            #     trial_name = self.args.get('log_dir_comment', f'default_patch').strip()
                            #     save_path = os.path.join(save_dir, f"{trial_name}_scene_{idx}.png")
                            #     vutils.save_image(self.patch.optmized_patch.data, save_path)

                return loss

            if self.args['update']=='lbfgs':
                optimizer.zero_grad()
                optimizer.step(closure)
                LR_decay.step()
            elif self.args['update']=='adam':
                optimizer.zero_grad()
                loss = closure()
                optimizer.step()
                del loss
                self.patch.optmized_patch.data.clamp_(0, 1)
            else: #bim
                loss = closure()
                grad = torch.autograd.grad(loss, [self.patch.optmized_patch] )[0]
                self.patch.optmized_patch = bim(grad, self.patch.optmized_patch,self.args['learning_rate'])
                del loss

        # return final_mrsr, final_ssim
        return final_e_blend, final_e_cover, final_ssim

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

                e_blend, e_cover = self.eval_core(batch_y[0], batch_y[1], target_depth, batch_resized[-1])
                
                adv_loss = self.adv_loss_fn(batch_y[0], target_depth, object_full_mask)
                
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

        # scene_idx = self.args.get('current_scene_idx', 0)
        # scene_name = self.args.get('current_scene_name', 'unknown')
        
        # save_base_dir = os.path.join("./saved_eval_images", f"scene_{scene_idx:03d}", f"epoch_{self.args['epoch']}")
        # os.makedirs(save_base_dir, exist_ok=True)
        
        # # 예: runs/2026-04-07_experiment/scene_000_...
        # scene_log_dir = self.log.log_dir 
        # experiment_root_dir = os.path.dirname(scene_log_dir) # runs/2026-04-07_experiment/
        scene_idx = self.args.get('current_scene_idx', 0)
        scene_name = self.args.get('current_scene_name', 'unknown')
        
        # 예: runs/2026-04-07_experiment/scene_000_...
        scene_log_dir = self.log.log_dir 
        experiment_root_dir = os.path.dirname(scene_log_dir) # runs/2026-04-07_experiment/
        
        experiment_name = os.path.basename(experiment_root_dir) 

        save_base_dir = os.path.join("./saved_eval_images", experiment_name, f"scene_{scene_idx:03d}", f"epoch_{self.args['epoch']}")
        os.makedirs(save_base_dir, exist_ok=True)


        with torch.no_grad():
            record = [[] for _ in range(2)]
            total_objs = len(self.objects.object_imgs_test[category]) - object_num + 1
            
            detail_rows = []

            sample_size = 5
            log_indices = random.sample(range(total_objs), sample_size)

            tqdm.write(f"▶ [Scene {scene_idx}] 전체 {total_objs}대 평가 중...")

            for idx in range(total_objs):
                current_file_full = self.objects.object_files_test[category][idx]
                current_file_name = current_file_full.split('.')[0] # 'car_38.jpg' -> 'car_38'


                batch, _ = self.env.accept_patch_and_objects(
                    True, patch, self.patch.mask, 
                    self.objects.object_imgs_test,
                    self.env.insert_range, insert_height, None, 
                    offset_patch=False, color_patch=False, offset_object=False, color_object=False,
                    object_idx_g=idx, 
                    category=category
                )

                batch = [F.interpolate(item, size=([int(self.args['input_height']), int(self.args['input_width'])])) for item in batch]
                batch_y = predict_batch(batch, MDE)

                # 점수 계산
                e_blend, e_cover = self.eval_core(batch_y[0], batch_y[1], batch_y[2], batch[-1])
                
                eb_val = e_blend.item()
                ec_val = e_cover.item()
                
                record[0].append(eb_val)
                record[1].append(ec_val)
                
                # 📝 개별 차량 데이터 기록
                detail_rows.append({
                    'Object_Idx': idx,
                    'Object_File': current_file_full,
                    'E_Blend': eb_val,
                    'E_Cover': ec_val
                })

                # 시각화 저장 로직 (선택된 5대만)
                if idx in log_indices and self.args['train_log_flag']:
                    # name_prefix = f"eval_{self.args['patch_file'][:-4]}_{category}_obj{idx}"
                    name_prefix = f"eval_{self.args['patch_file'][:-4]}_{current_file_name}"

                    # obj_dir = os.path.join(save_base_dir, f"obj_{idx}")
                    obj_dir = os.path.join(save_base_dir, f"obj_{current_file_name}")
                    os.makedirs(obj_dir, exist_ok=True)

                    vutils.save_image(batch[0][0], os.path.join(obj_dir, "adv_scene.png"))
                    vutils.save_image(batch[1][0], os.path.join(obj_dir, "ben_scene.png"))
                    vutils.save_image(batch[2][0], os.path.join(obj_dir, "target_scene.png"))
                    
                    colormap = plt.get_cmap('viridis')
                    def save_color_depth(depth_tensor, filename):
                        d_cpu = depth_tensor.detach().cpu().squeeze()
                        d_norm = (d_cpu - d_cpu.min()) / (d_cpu.max() - d_cpu.min() + 1e-7)
                        d_colored = colormap(d_norm.numpy())[..., :3]
                        d_tensor = torch.from_numpy(d_colored).permute(2, 0, 1)
                        vutils.save_image(d_tensor.float(), os.path.join(obj_dir, filename))

                    save_color_depth(batch_y[0][0], "adv_depth.png")
                    save_color_depth(batch_y[1][0], "ben_depth.png")
                    save_color_depth(batch_y[2][0], "target_depth.png")
                    
                    # log_img_eval(self.log, self.args['epoch'], name_prefix, [batch[0], batch[1], (patch*255).int().float()/255., batch_y[0], batch_y[1], batch[2], batch_y[2]])
                    
                    log_imgs_eval = [
                        batch[0].detach().cpu(), 
                        batch[1].detach().cpu(), 
                        ((patch*255).int().float()/255.).detach().cpu(), 
                        batch_y[0].detach().cpu(), 
                        batch_y[1].detach().cpu(), 
                        batch[2].detach().cpu(), 
                        batch_y[2].detach().cpu()
                    ]
                    log_img_eval(self.log, self.args['epoch'], name_prefix, log_imgs_eval)


            detail_df = pd.DataFrame(detail_rows)
            detail_csv_path = os.path.join(scene_log_dir, f"scene_{scene_idx:03d}_detail_results.csv")
            detail_df.to_csv(detail_csv_path, index=False)

            avg_eb = np.mean(record[0])
            avg_ec = np.mean(record[1])
            
            summary_csv_path = os.path.join(experiment_root_dir, "experiment_summary.csv")
            summary_row = pd.DataFrame([{
                'Scene_Idx': scene_idx,
                'Scene_Name': scene_name,
                'Avg_E_Blend': avg_eb,
                'Avg_E_Cover': avg_ec,
                'SSIM': self.args.get('current_ssim', 0),
                'Epoch': self.args['epoch']
            }])

            if not os.path.exists(summary_csv_path):
                summary_row.to_csv(summary_csv_path, index=False)
            else:
                summary_row.to_csv(summary_csv_path, mode='a', header=False, index=False)

            tqdm.write(f"💾 Scene {scene_idx:03d} 결과 저장 완료 (Mean E_Blend: {avg_eb:.4f})")

            return record
        

    def eval_core(self, adv_depth, ref_depth, pure_bg_depth, scene_obj_mask):
        if adv_depth.dim() == 3:
            B, H, W = adv_depth.shape
        else:
            B, C, H, W = adv_depth.shape
            
        # tau = 1 # 임계값 (real-world =5m)
        # tau = 10 # 임계값 (real-world =25m)
        # tau = 5 # 임계값 (real-world =12.5m)
        # tau = 1

        # 🚨 [수정] 절대값 기준(tau=10) 대신, 상대적 비율(%) 기준 적용
        # tau_ratio = 0.1      # 배경 깊이 대비 10% 이상 왜곡되면 동화(Blend)된 것으로 간주
        tau_ratio = 0.3      # 배경 깊이 대비 30% 이상 왜곡되면 동화(Blend)된 것으로 간주
        epsilon_ratio = 0.3 # 원본 차량 깊이 대비 30% 이상 깊이가 변했는지 확인
        
        e_blend_batch = 0.0
        e_cover_batch = 0.0
        
        for b in range(B):
            d_adv = adv_depth[b, 0] if adv_depth.dim() == 4 else adv_depth[b]
            d_ben = ref_depth[b, 0] if ref_depth.dim() == 4 else ref_depth[b]
            d_pure = pure_bg_depth[b, 0] if pure_bg_depth.dim() == 4 else pure_bg_depth[b]
            
            m = scene_obj_mask[b, 0] if scene_obj_mask.dim() == 4 else scene_obj_mask[b]
            
            if m.sum() == 0:
                continue
            
            # # 1. E_cover 계산 (adv vs benign) -> 패치가 차의 깊이를 얼마나 바꿨나?
            # cover_mask = (d_adv != d_ben) & (m > 0)

            # # epsilon = 0.2 # MDE모델의 오차 정도를 고려하기 위한 임계값
            # cover_mask = (torch.abs(d_adv - d_ben) > epsilon) & (m > 0)

            # e_cover = cover_mask.sum().float() / (m.sum() + 1e-7)
            # e_cover_batch += e_cover
            
            # # E_blend 계산 (adv vs pure_bg) -> 차가 진짜 배경에 얼마나 융화됐나?
            # target_mask = (m > 0).float()
            # row_bg_sum = (d_pure*target_mask).sum(dim=1)  
            # row_bg_count = target_mask.sum(dim=1)          
            
            # d_bg_bar = row_bg_sum / (row_bg_count + 1e-7) 
            # d_bg_map = d_bg_bar.unsqueeze(1).expand(H, W)
    
            # blend_mask = (torch.abs(d_adv - d_bg_map) > tau) & (m > 0)

            # e_blend = blend_mask.sum().float() / (m.sum() + 1e-7)
            # e_blend_batch += e_blend

            # 1. E_cover 계산 (비율 기반)
            # 패치 부착 후 차량의 깊이가 '기존 차량 깊이' 대비 얼마나(%) 변했는가?
            diff_cover_ratio = torch.abs(d_adv - d_ben) / (d_ben + 1e-7)
            cover_mask = (diff_cover_ratio > epsilon_ratio) & (m > 0)

            e_cover = cover_mask.sum().float() / (m.sum() + 1e-7)
            e_cover_batch += e_cover
            
            # 2. E_blend 계산 (비율 기반)
            target_mask = (m > 0).float()
            row_bg_sum = (d_pure * target_mask).sum(dim=1)  
            row_bg_count = target_mask.sum(dim=1)          
            
            d_bg_bar = row_bg_sum / (row_bg_count + 1e-7) 
            d_bg_map = d_bg_bar.unsqueeze(1).expand(H, W)
    
            # 공격 후 깊이가 '진짜 배경 깊이'와 비교했을 때 몇 % 차이나는가?
            # (이 차이가 tau_ratio보다 커야 공격 성공으로 간주)
            diff_blend_ratio = torch.abs(d_adv - d_bg_map) / (d_bg_map + 1e-7)
            
            # # [디버깅] 비율 오차가 실제로 어느 정도 찍히는지 확인
            # valid_ratios = diff_blend_ratio[m > 0]
            # if valid_ratios.numel() > 0:
            #     max_r = valid_ratios.max().item() * 100
            #     mean_r = valid_ratios.mean().item() * 100
            #     print(f"  [DEBUG] Blend 왜곡 비율 -> Max: {max_r:.2f}% | Mean: {mean_r:.2f}% (기준: {tau_ratio*100}%)")

            blend_mask = (diff_blend_ratio > tau_ratio) & (m > 0)
            e_blend = blend_mask.sum().float() / (m.sum() + 1e-7)
            e_blend_batch += e_blend

        e_blend_mean = (e_blend_batch / B).detach().clone().to(adv_depth.device)
        e_cover_mean = (e_cover_batch / B).detach().clone().to(adv_depth.device)
        
        return e_blend_mean, e_cover_mean