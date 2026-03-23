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
        # ✅ 오직 마스크(차량) 영역에 대해서만 adv_depth가 target_depth(순수 배경)와 같아지도록 L1 Loss 적용
        def adv_loss_fn(batch, batch_y, target_depth):
            _, _, _, _, object_full_mask = batch
            adv_depth = batch_y[0] 
            
            l1_loss = F.l1_loss(
                adv_depth * object_full_mask, 
                target_depth * object_full_mask, 
                reduction='sum'
            ) / (object_full_mask.sum() + 1e-7)
            
            return l1_loss

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

        # =========================================================================
        # ✅ 패치도, 차도 없는 "순수 쌩 배경 이미지" 및 깊이 맵 추출
        # =========================================================================
        clean_env = F.interpolate(self.env.env, size=([int(self.args['input_height']), int(self.args['input_width'])]))
        dummy_mask = torch.zeros_like(clean_env[:, 0:1, :, :])
        with torch.no_grad():
            bg_depth_res = predict_batch([clean_env, clean_env, clean_env, dummy_mask, dummy_mask], self.MDE)
            pure_bg_depth = bg_depth_res[0].detach() # 이게 완벽한 Clear Background Depth!
        # =========================================================================

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
                    
                    # 수치 디버깅 (Target vs Adv)
                    adv_diff = (batch_y[0] - target_depth) * object_full_mask
                    adv_bg_rmse = torch.sqrt(torch.sum(adv_diff**2) / (torch.sum(object_full_mask) + 1e-7))

                if epoch % 50 == 0 or (epoch+1) == self.args['epoch']:
                    tqdm.write(f"[Epoch {epoch:3d}] 원본 갭: {ben_bg_rmse.item():.4f} | 공격 후 갭: {adv_bg_rmse.item():.4f} ➔ 0 수렴 목표")
                    
                if self.args['train_log_flag']:
                    # self.log.add_scalar(f'{name_prefix}/debug/adv_bg_gap_rmse', adv_bg_rmse.item(), epoch)
                    # ✅ [NEW] 원본 갭도 텐서보드 스칼라에 기록
                    self.log.add_scalar(f'{name_prefix}/debug/benign_bg_gap_rmse', ben_bg_rmse.item(), epoch)
                    self.log.add_scalar(f'{name_prefix}/debug/adv_bg_gap_rmse', adv_bg_rmse.item(), epoch)
                
                mean_ori, mean_shift, max_shift, min_shift, arr =self.eval_core(batch_y[0], batch_y[1], batch_resized[-1])
                
                # ✅ 수정된 Loss (차량 마스크 영역만 순수 배경과 똑같아지도록)
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
                        # ✅ log.py로 넘겨줄 때 target_scene(clean_env)과 target_depth(pure_bg_depth)를 리스트에 추가!
                        log_img_train(self.log, epoch, name_prefix, [adv_scene_image, ben_scene_image, (self.patch.optmized_patch*255).int().float()/255., batch_y[0], batch_y[1], clean_env, target_depth])
                        self.log.add_image(f'{name_prefix}/train/object_mask', object_full_mask.detach().cpu()[0, 0], epoch, dataformats='HW')
                        log_scale_train(self.log,epoch, name_prefix, style_score, content_score, tv_score, adv_loss, mean_shift, max_shift, min_shift, mean_ori, arr)
                
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

        # =========================================================================
        # ✅ 순수 쌩 배경 이미지 및 깊이 추출 (Fixed Object 모드)
        # =========================================================================
        clean_env = F.interpolate(self.env.env, size=([int(self.args['input_height']), int(self.args['input_width'])]))
        dummy_mask = torch.zeros_like(clean_env[:, 0:1, :, :])
        with torch.no_grad():
            bg_depth_res = predict_batch([clean_env, clean_env, clean_env, dummy_mask, dummy_mask], self.MDE)
            pure_bg_depth = bg_depth_res[0].detach()
        # =========================================================================

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

                mean_ori, mean_shift, max_shift, min_shift, arr =self.eval_core(batch_y[0], batch_y[1], batch_resized[-1])
                
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
                        # ✅ log.py로 넘겨줄 때 target_scene(clean_env)과 target_depth(pure_bg_depth) 추가
                        log_img_train(self.log, epoch, name_prefix, [adv_scene_image, ben_scene_image, (self.patch.optmized_patch*255).int().float()/255., batch_y[0], batch_y[1], clean_env, target_depth])
                        log_scale_train(self.log,epoch, name_prefix, style_score, content_score, tv_score, adv_loss, mean_shift, max_shift, min_shift, mean_ori, arr)
                
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

    #     with torch.no_grad():
    #         record = [[] for _ in range(5)]
    #         for idx in [0]:
    #             batch, _ = self.env.accept_patch_and_objects(True, patch, self.patch.mask, self.objects.object_imgs_train, self.env.insert_range, insert_height, None, offset_patch=False, color_patch=False, offset_object=False, color_object=False,object_idx_g=idx, category=category)
                
    #             batch= [ F.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch]
    #             batch_y= predict_batch(batch, MDE)
                
    #             # 테스트 로그 기록 시에는 target 이미지가 없으므로 기존 방식 유지 (길이 5로 전송)
    #             if idx == 0 and self.args['train_log_flag']:
    #                 name_prefix = f"eval_{self.args['patch_file'][:-4]}_{category}"
    #                 log_img_train(self.log, self.args['epoch'], name_prefix, [batch[0], batch[1], (patch*255).int().float()/255., batch_y[0], batch_y[1], batch[2], batch_y[1]])

    #             mean_ori, mean_shift, max_shift, min_shift, arr =self.eval_core(batch_y[0], batch_y[1], batch[-1])
    #             record[0].append(mean_shift.item())
    #             record[1].append(max_shift.item())
    #             record[2].append(min_shift.item())
    #             record[3].append((mean_shift/mean_ori).item())
    #             record[4].append(arr.item())
    #         return record

    # def eval(self, MDE, category, insert_height=None, patch=None):
    #     if patch is None:
    #         patch = self.patch.optmized_patch
    #     if self.args['test_quan_patch_flag']:
    #         patch = (patch*255).int().float()/255.
        
    #     # category = 'car' (이 부분은 인자로 받으므로 주석 처리하거나 지워도 됨)

    #     if category == 'pas':
    #          object_num=3
    #     else:
    #         object_num=1

    #     with torch.no_grad():
    #         record = [[] for _ in range(5)]
            
    #         # ✅ [수정 1] [0]으로 고정했던 걸 다시 60대(전체 테스트셋) 순회로 변경
    #         for idx in range(len(self.objects.object_imgs_test[category])-object_num+1):
                
    #             # ✅ [수정 2] object_imgs_train -> object_imgs_test 로 변경
    #             # ✅ [수정 3] object_idx_g=0 -> object_idx_g=idx 로 변경
    #             batch, _ = self.env.accept_patch_and_objects(
    #                 True, patch, self.patch.mask, 
    #                 self.objects.object_imgs_test, # 테스트 이미지 폴더 사용!
    #                 self.env.insert_range, insert_height, None, 
    #                 offset_patch=self.args['test_offset_patch_flag'], 
    #                 color_patch=self.args['test_color_patch_flag'], 
    #                 offset_object=self.args['test_offset_object_flag'], 
    #                 color_object=self.args['test_color_object_flag'],
    #                 object_idx_g=idx, # 매번 다른 차 불러오기!
    #                 category=category
    #             )
                
    #             batch = [ F.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch]
    #             batch_y = predict_batch(batch, MDE)
                
    #             # ✅ 테스트 셋의 첫 번째 차(idx==0) 결과만 텐서보드에 시각화 기록
    #             if idx == 0 and self.args['train_log_flag']:
    #                 # 텐서보드에 'eval_6_car' 라는 새로운 태그 그룹으로 저장됨!
    #                 name_prefix = f"eval_{self.args['patch_file'][:-4]}_{category}"
    #                 # 테스트엔 타겟 이미지가 없으니, 에러 방지용으로 원본(batch_y[1])을 두 번 넣음
    #                 log_img_train(self.log, self.args['epoch'], name_prefix, [batch[0], batch[1], (patch*255).int().float()/255., batch_y[0], batch_y[1], batch[2], batch_y[1]])

    #             mean_ori, mean_shift, max_shift, min_shift, arr = self.eval_core(batch_y[0], batch_y[1], batch[-1])
    #             record[0].append(mean_shift.item())
    #             record[1].append(max_shift.item())
    #             record[2].append(min_shift.item())
    #             record[3].append((mean_shift/mean_ori).item())
    #             record[4].append(arr.item())
                
    #         return record

    def eval(self, MDE, category, insert_height=None, patch=None):
     if patch is None:
         patch = self.patch.optmized_patch
     if self.args['test_quan_patch_flag']:
         patch = (patch*255).int().float()/255.

     category = 'car'
     object_num = 1

     with torch.no_grad():
         record = [[] for _ in range(5)]

         # ✅ [수정 1] for idx in [0] 지우고, Test Set 전체(60개)를 순회하도록 복구
         for idx in range(len(self.objects.object_imgs_test[category]) - object_num + 1):

             # ✅ [수정 2] object_imgs_train -> object_imgs_test 로 변경 (진짜 낯선 테스트 객체!)
             # ✅ [수정 3] object_idx_g=0 -> object_idx_g=idx 로 변경
             batch, _ = self.env.accept_patch_and_objects(
                 True, patch, self.patch.mask, 
                 self.objects.object_imgs_test, # 여기서 Test Set 폴더를 연결!
                 self.env.insert_range, insert_height, None, 
                 offset_patch=False, color_patch=False, offset_object=False, color_object=False,
                 object_idx_g=idx, # 0번이 아니라 매번 새로운 차 인덱스 부여
                 category=category
             )

             batch= [ F.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch]
             batch_y= predict_batch(batch, MDE)

             # ✅ [수정 4] Test Set의 첫 번째 차(idx==0) 결과만 텐서보드에 시각화
             # 새로 만든 log_img_eval 함수를 호출해서 /eval/ 탭으로 예쁘게 분리!
             if idx == 0 and self.args['train_log_flag']:
                 name_prefix = f"eval_{self.args['patch_file'][:-4]}_{category}"
                 log_img_eval(self.log, self.args['epoch'], name_prefix, [batch[0], batch[1], (patch*255).int().float()/255., batch_y[0], batch_y[1], batch[2], batch_y[1]])

             mean_ori, mean_shift, max_shift, min_shift, arr =self.eval_core(batch_y[0], batch_y[1], batch[-1])
             record[0].append(mean_shift.item())
             record[1].append(max_shift.item())
             record[2].append(min_shift.item())
             record[3].append((mean_shift/mean_ori).item())
             record[4].append(arr.item())

         return record

    def eval_core(self, adv_depth, ref_depth, scene_obj_mask):
        shift=(adv_depth-ref_depth)*scene_obj_mask
        mean_shift = torch.sum(shift)/torch.sum(scene_obj_mask)
        mean_ori = torch.sum(ref_depth*scene_obj_mask)/torch.sum(scene_obj_mask)
        max_shift=shift[scene_obj_mask==1].max()
        min_shift=shift[scene_obj_mask==1].min()

        shift = (adv_depth - ref_depth)
        relative_shift = (shift / ref_depth)
        affect_region = relative_shift > 0.14
        affect_region = affect_region * scene_obj_mask[:, 0, :, :]
        arr = affect_region.sum() / scene_obj_mask[:, 0, :, :].sum()
        return mean_ori, mean_shift, max_shift, min_shift, arr