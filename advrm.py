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
            if d_adv.dim() == 3: d_adv = d_adv.unsqueeze(1)
            if d_bg.dim() == 3: d_bg = d_bg.unsqueeze(1)
            if m.dim() == 3: m = m.unsqueeze(1)
            
            mask_float = (m > 0).float()
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
        
                # e_blend, e_cover = self.eval_core(batch_y[0], batch_y[1], target_depth, batch_resized[-1])
                
                # adv_loss = self.adv_loss_fn(batch_y[0], target_depth, object_full_mask)

                # 다중 Threshold 딕셔너리를 반환받으므로, 학습 로그 기록용으로는 0.15(기본값)만 꺼내서 사용!
                e_blend_result, e_cover = self.eval_core(batch_y[0], batch_y[1], target_depth, batch_resized[-1])

                # 딕셔너리인지 확인 후, 0.15 임계값의 결과만 단일 텐서로 추출
                e_blend = e_blend_result[0.15] if isinstance(e_blend_result, dict) else e_blend_result

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
                        log_scale_train(self.log, epoch, name_prefix, style_score, content_score, tv_score, adv_loss, e_blend, e_cover)

                
                if self.args['inner_eval_flag']:
                    if (epoch+1)==self.args['epoch']: 
                        patch_opt = self.patch.optmized_patch.detach().cpu().squeeze().permute(1, 2, 0).numpy()
                        patch_init = self.patch.init_patch.detach().cpu().squeeze().permute(1, 2, 0).numpy()
                        final_ssim = ssim_metric(patch_init, patch_opt, data_range=1.0, channel_axis=-1)
                        self.args['current_ssim'] = final_ssim  # CSV 저장할 때 사용

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

                            final_e_blend = np.mean(record[0])
                            final_e_cover = np.mean(record[1])

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

    # def eval(self, MDE, category, insert_height=None, patch=None):
    #     if patch is None:
    #         patch = self.patch.optmized_patch
    #     if self.args['test_quan_patch_flag']:
    #         patch = (patch*255).int().float()/255.

    #     category = 'car'
    #     object_num = 1

    #     scene_idx = self.args.get('current_scene_idx', 0)
    #     scene_name = self.args.get('current_scene_name', 'unknown')
        
    #     scene_log_dir = self.log.log_dir 
    #     experiment_root_dir = os.path.dirname(scene_log_dir) 
    #     experiment_name = os.path.basename(experiment_root_dir) 

    #     save_base_dir = os.path.join("./saved_eval_images", experiment_name, f"scene_{scene_idx:03d}", f"epoch_{self.args['epoch']}")
    #     os.makedirs(save_base_dir, exist_ok=True)

    #     with torch.no_grad():
    #         record = [[] for _ in range(2)]
    #         total_objs = len(self.objects.object_imgs_test[category]) - object_num + 1
        
    #         detail_rows = []

    #         all_saved_data = [] 
            
    #         tqdm.write(f"▶ [Scene {scene_idx}] 전체 {total_objs}대 평가 중...")

    #         for idx in range(total_objs):
    #             current_file_full = self.objects.object_files_test[category][idx]
                
    #             batch, _ = self.env.accept_patch_and_objects(
    #                 True, patch, self.patch.mask, 
    #                 self.objects.object_imgs_test,
    #                 self.env.insert_range, insert_height, None, 
    #                 offset_patch=False, color_patch=False, offset_object=False, color_object=False,
    #                 object_idx_g=idx, 
    #                 category=category
    #             )

    #             batch = [F.interpolate(item, size=([int(self.args['input_height']), int(self.args['input_width'])])) for item in batch]
    #             batch_y = predict_batch(batch, MDE)

    #             e_blend, e_cover = self.eval_core(batch_y[0], batch_y[1], batch_y[2], batch[-1])
                
    #             eb_val = e_blend.item()
    #             ec_val = e_cover.item()
                
    #             record[0].append(eb_val)
    #             record[1].append(ec_val)
                
    #             detail_rows.append({
    #                 'Object_Idx': idx,
    #                 'Object_File': current_file_full,
    #                 'Pixel_Offset': px_offset,
    #                 'E_Blend': eb_val,
    #                 'E_Cover': ec_val
    #             })

    #             if self.args['train_log_flag']:
    #                 saved_batch = [b.detach().cpu() for b in batch]
    #                 saved_batch_y = [by.detach().cpu() for by in batch_y]
    #                 all_saved_data.append((saved_batch, saved_batch_y))

    #         avg_eb = np.mean(record[0])
    #         avg_ec = np.mean(record[1])
    #         min_eb = np.min(record[0])
    #         max_eb = np.max(record[0])

    #         # CSV 맨 밑에 통계값 추가
    #         detail_rows.append({'Object_Idx': 'MIN', 'Object_File': '-', 'E_Blend': min_eb, 'E_Cover': np.min(record[1])})
    #         detail_rows.append({'Object_Idx': 'MAX', 'Object_File': '-', 'E_Blend': max_eb, 'E_Cover': np.max(record[1])})
    #         detail_rows.append({'Object_Idx': 'TOTAL_AVG', 'Object_File': '-', 'E_Blend': avg_eb, 'E_Cover': avg_ec})
            
    #         # 전체 이미지 저장 로직 (Top/Bottom 10 필터링 제거)
    #         if self.args['train_log_flag'] and total_objs > 0:
    #             target_indices = range(total_objs) # 전체 차량 인덱스 대상
                
    #             for idx in target_indices:
    #                 best_file_full = self.objects.object_files_test[category][idx]
    #                 current_file_name = best_file_full.split('.')[0]
    #                 name_prefix = f"eval_{self.args['patch_file'][:-4]}_{current_file_name}"

    #                 target_batch, target_batch_y = all_saved_data[idx]
                    
    #                 obj_dir = os.path.join(save_base_dir, f"obj_{current_file_name}")
    #                 os.makedirs(obj_dir, exist_ok=True)

    #                 vutils.save_image(target_batch[0][0], os.path.join(obj_dir, "adv_scene.png"))
    #                 vutils.save_image(target_batch[1][0], os.path.join(obj_dir, "ben_scene.png"))
    #                 vutils.save_image(target_batch[2][0], os.path.join(obj_dir, "target_scene.png"))
                    
    #                 colormap = plt.get_cmap('viridis')
    #                 def save_color_depth(depth_tensor, filename):
    #                     d_cpu = depth_tensor.squeeze() 
    #                     d_norm = (d_cpu - d_cpu.min()) / (d_cpu.max() - d_cpu.min() + 1e-7)
    #                     d_colored = colormap(d_norm.numpy())[..., :3]
    #                     d_tensor = torch.from_numpy(d_colored).permute(2, 0, 1)
    #                     vutils.save_image(d_tensor.float(), os.path.join(obj_dir, filename))

    #                 save_color_depth(target_batch_y[0][0], "adv_depth.png")
    #                 save_color_depth(target_batch_y[1][0], "ben_depth.png")
    #                 save_color_depth(target_batch_y[2][0], "target_depth.png")
                    
    #                 log_imgs_eval = [
    #                     target_batch[0], target_batch[1], ((patch*255).int().float()/255.).detach().cpu(), 
    #                     target_batch_y[0], target_batch_y[1], target_batch[2], target_batch_y[2]
    #                 ]
    #                 log_img_eval(self.log, self.args['epoch'], name_prefix, log_imgs_eval)
                
    #             tqdm.write(f"✨ [Saved] Scene {scene_idx}의 전체 차량 (총 {len(target_indices)}대) 6장 세트 이미지 저장 완료")

    #         detail_df = pd.DataFrame(detail_rows)
    #         detail_csv_path = os.path.join(scene_log_dir, f"scene_{scene_idx:03d}_detail_results.csv")
    #         detail_df.to_csv(detail_csv_path, index=False)

    #         summary_csv_path = os.path.join(experiment_root_dir, "experiment_summary.csv")
    #         summary_row = pd.DataFrame([{
    #             'Scene_Idx': scene_idx,
    #             'Scene_Name': scene_name,
    #             'Avg_E_Blend': avg_eb,
    #             'Avg_E_Cover': avg_ec,
    #             'SSIM': self.args.get('current_ssim', 0),
    #             'Epoch': self.args['epoch']
    #         }])

    #         if not os.path.exists(summary_csv_path):
    #             summary_row.to_csv(summary_csv_path, index=False)
    #         else:
    #             summary_row.to_csv(summary_csv_path, mode='a', header=False, index=False)

    #         return record

    def eval(self, MDE, category, insert_height=None, patch=None):
        if patch is None:
            patch = self.patch.optmized_patch
        if self.args['test_quan_patch_flag']:
            patch = (patch*255).int().float()/255.

        category = 'car'
        object_num = 1

        scene_idx = self.args.get('current_scene_idx', 0)
        scene_name = self.args.get('current_scene_name', 'unknown')
        
        scene_log_dir = self.log.log_dir 
        experiment_root_dir = os.path.dirname(scene_log_dir) 
        experiment_name = os.path.basename(experiment_root_dir) 

        save_base_dir = os.path.join("./saved_eval_images", experiment_name, f"scene_{scene_idx:03d}", f"epoch_{self.args.get('epoch', 'test')}")
        os.makedirs(save_base_dir, exist_ok=True)

        with torch.no_grad():
            total_objs = len(self.objects.object_imgs_test[category]) - object_num + 1
            detail_rows = []
            
            # 💡 [추가됨] 논문 그래프를 위한 고정 픽셀 오프셋 리스트 (Upward)
            record = [[] for _ in range(2)]
            offsets_to_test = [0, 5, 10, 15, 20]

            
            
            tqdm.write(f"▶ [Scene {scene_idx}] {total_objs}대 차량 × {len(offsets_to_test)}개 오프셋 정밀 평가 중...")

            for idx in range(total_objs):
                current_file_full = self.objects.object_files_test[category][idx]
                current_file_name = current_file_full.split('.')[0]
                
                for px_offset in offsets_to_test:
                    # 픽셀을 비율(v)로 변환 (Upward = Y좌표 감소 = 음수)
                    v_ratio = -px_offset / self.args['patch_height']
                    self.args['test_v_ratio'] = v_ratio 
                    
                    batch, _ = self.env.accept_patch_and_objects(
                        True, patch, self.patch.mask, 
                        self.objects.object_imgs_test,
                        self.env.insert_range, insert_height, None, 
                        offset_patch=False, color_patch=False, offset_object=False, color_object=False,
                        object_idx_g=idx, 
                        category=category
                    )

                    batch_resized = [F.interpolate(item, size=([int(self.args['input_height']), int(self.args['input_width'])])) for item in batch]
                    batch_y = predict_batch(batch_resized, MDE)

                    # # 💡 [추가됨] 고급 지표 (RMSE, Area, Variance) 계산
                    # m = batch_resized[-1]
                    # target_depth = batch_y[2]
                    
                    # obj_pixel_area = m.sum().item()
                    # valid_bg = target_depth[0, 0][m[0, 0] > 0]
                    # bg_variance = valid_bg.var().item() if valid_bg.numel() > 1 else 0.0

                    # 💡 [추가됨] 고급 지표 (RMSE, Area, Variance) 계산
                    m = batch_resized[-1]
                    target_depth = batch_y[2]
                    
                    # 마스크와 깊이맵의 차원(3D vs 4D) 호환성을 완벽하게 맞춤
                    m_2d = m[0, 0] if m.dim() == 4 else m[0]
                    td_2d = target_depth[0, 0] if target_depth.dim() == 4 else target_depth[0]
                    
                    obj_pixel_area = m.sum().item()
                    valid_bg = td_2d[m_2d > 0]
                    bg_variance = valid_bg.var().item() if valid_bg.numel() > 1 else 0.0

                    
                    diff = (batch_y[0] - target_depth) * m
                    adv_bg_rmse = torch.sqrt(torch.sum(diff**2) / (torch.sum(m) + 1e-7)).item()

                    # 다중 Threshold E_blend 딕셔너리 받아오기
                    e_blend_dict, e_cover = self.eval_core(batch_y[0], batch_y[1], target_depth, m)

                    # 💡 [추가할 부분 2] 기준 Threshold(0.15)의 값을 record에 저장
                    record[0].append(e_blend_dict[0.15].item())
                    record[1].append(e_cover.item())
                    
                    # CSV에 저장할 데이터 구성
                    row_data = {
                        'Object_Idx': idx,
                        'Object_File': current_file_full,
                        'Pixel_Offset': px_offset,
                        'Object_Area': obj_pixel_area,
                        'BG_Variance': bg_variance,
                        'RMSE_Error': adv_bg_rmse,
                        'E_Cover': e_cover.item()
                    }
                    
                    # 다중 Threshold 값 전개
                    for tau, val in e_blend_dict.items():
                        row_data[f'E_Blend_{int(tau*100):02d}'] = val.item()
                        
                    detail_rows.append(row_data)

                    # 💡 [메모리 최적화] 이미지를 리스트에 모으지 않고 즉시 저장
                    if self.args.get('train_log_flag', True):
                        # 객체와 오프셋별로 폴더 분리
                        obj_dir = os.path.join(save_base_dir, f"obj_{current_file_name}", f"offset_{px_offset}")
                        os.makedirs(obj_dir, exist_ok=True)

                        vutils.save_image(batch_resized[0][0].detach().cpu(), os.path.join(obj_dir, "adv_scene.png"))
                        vutils.save_image(batch_resized[1][0].detach().cpu(), os.path.join(obj_dir, "ben_scene.png"))
                        vutils.save_image(batch_resized[2][0].detach().cpu(), os.path.join(obj_dir, "target_scene.png"))
                        
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

            # 전체 CSV 저장
            detail_df = pd.DataFrame(detail_rows)
            detail_csv_path = os.path.join(scene_log_dir, f"scene_{scene_idx:03d}_detail_results.csv")
            detail_df.to_csv(detail_csv_path, index=False)
            tqdm.write(f"✨ [Saved] Scene {scene_idx} 상세 평가 데이터 기록 완료 (경로: {detail_csv_path})")

            # (선택) 이전 코드에 있던 summary_csv_path 저장 로직은 필요하다면 동일하게 유지
            
            return record # 평가만 하므로 record 리턴 불필요

    # def eval_core(self, adv_depth, ref_depth, pure_bg_depth, scene_obj_mask):
    #     if adv_depth.dim() == 3:
    #         B, H, W = adv_depth.shape
    #     else:
    #         B, C, H, W = adv_depth.shape
            
    #     tau_ratio = 0.15     
    #     tau_scale = 0.05     
    #     epsilon_ratio = 0.3  
        
    #     e_blend_batch = 0.0
    #     e_cover_batch = 0.0
        
    #     for b in range(B):
    #         d_adv = adv_depth[b, 0] if adv_depth.dim() == 4 else adv_depth[b]
    #         d_ben = ref_depth[b, 0] if ref_depth.dim() == 4 else ref_depth[b]
    #         d_pure = pure_bg_depth[b, 0] if pure_bg_depth.dim() == 4 else pure_bg_depth[b]
            
    #         m = scene_obj_mask[b, 0] if scene_obj_mask.dim() == 4 else scene_obj_mask[b]
            
    #         if m.sum() == 0:
    #             continue

    #         diff_cover_ratio = torch.abs(d_adv - d_ben) / (d_ben + 1e-7)
    #         cover_mask = (diff_cover_ratio > epsilon_ratio) & (m > 0)

    #         e_cover = cover_mask.sum().float() / (m.sum() + 1e-7)
    #         e_cover_batch += e_cover
            
    #         target_mask = (m > 0).float()
    #         row_bg_sum = (d_pure * target_mask).sum(dim=1)  
    #         row_bg_count = target_mask.sum(dim=1)          
            
    #         d_bg_bar = row_bg_sum / (row_bg_count + 1e-7) 
    #         d_bg_map = d_bg_bar.unsqueeze(1).expand(H, W)
    
    #         diff_blend_ratio = torch.abs(d_adv - d_bg_map) / (d_bg_map + 1e-7)
    #         # print(diff_blend_ratio) # 필요시 주석 해제

    #         d_pure_flat = d_pure.reshape(-1)
    #         d_max_robust = torch.quantile(d_pure_flat, 0.80)
    #         # print("d_max_robust: ", d_max_robust.item()) 
    #         d_min_robust = torch.quantile(d_pure_flat, 0.20)
    #         scene_depth_range = d_max_robust - d_min_robust + 1e-7
    #         diff_absolute_ratio = torch.abs(d_adv - d_bg_map) / scene_depth_range

    #         blend_mask = (diff_blend_ratio > tau_ratio) & (diff_absolute_ratio > tau_scale) & (m > 0)
            
    #         e_blend = blend_mask.sum().float() / (m.sum() + 1e-7)
    #         e_blend_batch += e_blend

    #     e_blend_mean = (e_blend_batch / B).detach().clone().to(adv_depth.device)
    #     e_cover_mean = (e_cover_batch / B).detach().clone().to(adv_depth.device)
        
    #     return e_blend_mean, e_cover_mean

    # def eval_core(self, adv_depth, ref_depth, pure_bg_depth, scene_obj_mask):
    #     if adv_depth.dim() == 3:
    #         B, H, W = adv_depth.shape
    #     else:
    #         B, C, H, W = adv_depth.shape
            
    #     tau_scale = 0.05     
    #     epsilon_ratio = 0.3  
        
    #     # 💡 [수정됨] 논문 방어력 붕괴 곡선을 위한 다중 Threshold 리스트
    #     tau_ratios = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    #     e_blend_batch = {tau: 0.0 for tau in tau_ratios}
    #     e_cover_batch = 0.0
        
    #     for b in range(B):
    #         d_adv = adv_depth[b, 0] if adv_depth.dim() == 4 else adv_depth[b]
    #         d_ben = ref_depth[b, 0] if ref_depth.dim() == 4 else ref_depth[b]
    #         d_pure = pure_bg_depth[b, 0] if pure_bg_depth.dim() == 4 else pure_bg_depth[b]
    #         m = scene_obj_mask[b, 0] if scene_obj_mask.dim() == 4 else scene_obj_mask[b]
            
    #         if m.sum() == 0: continue

    #         # E_cover 계산
    #         diff_cover_ratio = torch.abs(d_adv - d_ben) / (d_ben + 1e-7)
    #         cover_mask = (diff_cover_ratio > epsilon_ratio) & (m > 0)
    #         e_cover_batch += cover_mask.sum().float() / (m.sum() + 1e-7)
            
    #         # E_blend 계산을 위한 배경 스케일 분석
    #         target_mask = (m > 0).float()
    #         row_bg_sum = (d_pure * target_mask).sum(dim=1)  
    #         row_bg_count = target_mask.sum(dim=1)          
    #         d_bg_map = (row_bg_sum / (row_bg_count + 1e-7)).unsqueeze(1).expand(H, W)
    
    #         diff_blend_ratio = torch.abs(d_adv - d_bg_map) / (d_bg_map + 1e-7)

    #         d_pure_flat = d_pure.reshape(-1)
    #         d_max_robust = torch.quantile(d_pure_flat, 0.80)
    #         d_min_robust = torch.quantile(d_pure_flat, 0.20)
    #         scene_depth_range = d_max_robust - d_min_robust + 1e-7
    #         diff_absolute_ratio = torch.abs(d_adv - d_bg_map) / scene_depth_range

    #         # 💡 [수정됨] 여러 Threshold에 대해 동시에 E_blend 계산
    #         for tau in tau_ratios:
    #             blend_mask = (diff_blend_ratio > tau) & (diff_absolute_ratio > tau_scale) & (m > 0)
    #             e_blend_batch[tau] += blend_mask.sum().float() / (m.sum() + 1e-7)

    #     # 딕셔너리 형태로 평균 반환
    #     e_blend_mean = {tau: (val / B).detach().clone().to(adv_depth.device) for tau, val in e_blend_batch.items()}
    #     e_cover_mean = (e_cover_batch / B).detach().clone().to(adv_depth.device)
        
    #     return e_blend_mean, e_cover_mean

    def eval_core(self, adv_depth, ref_depth, pure_bg_depth, scene_obj_mask):
        if adv_depth.dim() == 3:
            B, H, W = adv_depth.shape
        else:
            B, C, H, W = adv_depth.shape
            
        # Z-score 임계값: 해당 씬의 표준편차(1 Sigma) 대비 몇 배 이상 튀어야 공격 성공으로 볼 것인가?
        # 기존 tau_scale(0.05) 대신, Z-score에 맞는 적절한 임계값 설정 (예: 0.5 -> 0.5 표준편차 이상)
        z_score_threshold = 0.5     
        epsilon_ratio = 0.3  
        
        # 다중 Threshold 리스트 (비율 기준)
        tau_ratios = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
        e_blend_batch = {tau: 0.0 for tau in tau_ratios}
        e_cover_batch = 0.0
        
        for b in range(B):
            d_adv = adv_depth[b, 0] if adv_depth.dim() == 4 else adv_depth[b]
            d_ben = ref_depth[b, 0] if ref_depth.dim() == 4 else ref_depth[b]
            d_pure = pure_bg_depth[b, 0] if pure_bg_depth.dim() == 4 else pure_bg_depth[b]
            m = scene_obj_mask[b, 0] if scene_obj_mask.dim() == 4 else scene_obj_mask[b]
            
            if m.sum() == 0: continue

            # 1. E_cover 계산 (변함 없음)
            diff_cover_ratio = torch.abs(d_adv - d_ben) / (d_ben + 1e-7)
            cover_mask = (diff_cover_ratio > epsilon_ratio) & (m > 0)
            e_cover_batch += cover_mask.sum().float() / (m.sum() + 1e-7)
            
            # 2. E_blend 계산을 위한 배경 스케일 분석
            target_mask = (m > 0).float()
            row_bg_sum = (d_pure * target_mask).sum(dim=1)  
            row_bg_count = target_mask.sum(dim=1)          
            d_bg_map = (row_bg_sum / (row_bg_count + 1e-7)).unsqueeze(1).expand(H, W)
    
            # Local 비율 오차 (기존 유지: 객체 주변부 대비 얼마나 튀는가)
            diff_blend_ratio = torch.abs(d_adv - d_bg_map) / (d_bg_map + 1e-7)

            # 💡 [수정됨] Global Z-Score 정규화 오차 계산
            # 해당 씬 전체 깊이의 표준편차(Sigma)를 구함
            scene_bg_std = torch.std(d_pure.float()) + 1e-7 
            
            # 절대 오차를 표준편차로 나누어 Z-Score로 변환
            # 의미: "이 픽셀의 오차가 씬 전체 깊이 편차의 몇 배인가?"
            diff_z_score = torch.abs(d_adv - d_bg_map) / scene_bg_std

            # 💡 여러 Threshold에 대해 동시에 E_blend 계산
            for tau in tau_ratios:
                # 공격 성공 조건: 
                # 1) 로컬 비율(tau) 이상 왜곡되었고, 
                # 2) 씬 전체의 Z-score 기준(z_score_threshold)으로도 충분히 의미 있게 튀었을 때
                blend_mask = (diff_blend_ratio > tau) & (diff_z_score > z_score_threshold) & (m > 0)
                e_blend_batch[tau] += blend_mask.sum().float() / (m.sum() + 1e-7)

        # 딕셔너리 형태로 평균 반환
        e_blend_mean = {tau: (val / B).detach().clone().to(adv_depth.device) for tau, val in e_blend_batch.items()}
        e_cover_mean = (e_cover_batch / B).detach().clone().to(adv_depth.device)
        
        return e_blend_mean, e_cover_mean