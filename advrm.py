from images import *
from load_model import load_target_model, integrated_mde_models, predict_batch, predict_depth_fn
import pandas as pd
import torchvision.models as models
from model import get_style_model_and_losses
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from log import log_img_train,log_scale_train,log_scale_eval
import urllib
import traceback
from torch.optim import Adam
import torch.optim as optim
from lr_decay import PolynomialLRDecay
import os
from torchvision.utils import save_image

from skimage.metrics import structural_similarity as compare_ssim



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
            # self.objects.load_object_mask(self.args['obj_full_mask_file'], self.args['obj_full_mask_dir'])
            self.run = self.run_with_fixed_object
        self.MDE = load_target_model(self.args['depth_model'], self.args)
        self.configure_loss(self.patch)

    def configure_loss(self, patch):
        def adv_loss_fn(batch, batch_y):
            adv_score = 0
            _, _, _, patch_full_mask, object_full_mask = batch
            adv_depth, ben_depth, _, tar_depth = batch_y
            object_depth = adv_depth * object_full_mask
            object_diff_ben = (adv_depth-ben_depth)*object_full_mask
            patch_full_mask = torch.clip(patch_full_mask - object_full_mask, 0, 1)
            patch_diff_tar= torch.abs((adv_depth-ben_depth)*patch_full_mask)
            adv_score += -object_depth.sum()/(object_full_mask.sum()+1e-7) \
                - object_depth[object_diff_ben<0].sum()/(object_full_mask[object_diff_ben<0].sum()+1e-7) \
                + patch_diff_tar.sum() / (patch_full_mask.sum()+1e-7)       
            return adv_score
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

    # def run_with_random_object(self, scene_dir, scene_file, idx, points):
    #     self.env = ENV(self.args, scene_file, scene_dir, idx, points)
    #     name_prefix = f"{self.args['patch_file'][:-4]}_{scene_file[:-4]}_{idx}"
    #     if self.args['update']=='lbfgs':
    #         optimizer = optim.LBFGS([self.patch.optmized_patch], lr=self.args['learning_rate'])
    #         LR_decay = PolynomialLRDecay(optimizer, self.args['epoch']//2, self.args['learning_rate']/2, 0.9)
            
    #     # --- [추가]: Best Loss를 기록할 변수 초기화 ---
    #     best_loss = float('inf')
    #     best_eval_mrsr = -float('inf')
    #     best_patch = None
    #     best_epoch = 0 

    #     for epoch in tqdm(range(self.args['epoch']), desc=f"Training {idx}/{self.args['scene_num']}"):
    #         def closure(): 
    #             self.patch.optmized_patch.data.clamp_(0, 1)
    #             batch, patch_size = self.env.accept_patch_and_objects(False, self.patch.optmized_patch, self.patch.mask, self.objects.object_imgs_train, self.env.insert_range, None, None, offset_patch=self.args['train_offset_patch_flag'], color_patch=self.args['train_color_patch_flag'], offset_object=self.args['train_offset_object_flag'], color_object=self.args['train_color_object_flag'])

    #             adv_scene_image, ben_scene_image, scene_img, patch_full_mask, object_full_mask = batch
                
    #             # ==== [추가 시작]: 패치 & 장애물 삽입 확인용 이미지 저장 ====
    #             # 너무 많이 저장되지 않도록 첫 에폭(0)과 100 단위 에폭에서만 저장합니다.
    #             if epoch == 0 or (epoch + 1) % 100 == 0:
    #                 safe_name_prefix = name_prefix.replace('/', '_').replace('\\', '_')
    #                 debug_dir = os.path.join(self.args.get('log_dir', 'runs'), "debug_scene")
    #                 os.makedirs(debug_dir, exist_ok=True)
                    
    #                 # 1. 패치와 장애물이 모두 합성된 실제 공격 씬 이미지
    #                 save_image(adv_scene_image.clone().detach().cpu(), 
    #                            os.path.join(debug_dir, f"{safe_name_prefix}_ep{epoch+1}_scene.png"))
                    
    #                 # 2. 합성 위치를 정확히 확인할 수 있는 마스크 이미지 (디버깅용)
    #                 save_image(patch_full_mask.clone().detach().cpu(), 
    #                            os.path.join(debug_dir, f"{safe_name_prefix}_ep{epoch+1}_patch_mask.png"))
    #                 save_image(object_full_mask.clone().detach().cpu(), 
    #                            os.path.join(debug_dir, f"{safe_name_prefix}_ep{epoch+1}_object_mask.png"))
    #             # ==== [추가 끝] ====
                
    #             batch= [ torch.nn.functional.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch]
    #             batch_y = predict_batch(batch, self.MDE)
                
    #             mean_ori, mean_shift, max_shift, min_shift, arr =self.eval_core(batch_y[0], batch_y[1], batch[-1])
                
    #             # 👈 [추가] 현재 10단위 출력을 위해 현재 배치의 Train MRSR을 계산해둡니다.
    #             self.current_train_mrsr = (mean_shift / mean_ori).item() if mean_ori.item() != 0 else 0.0
                
    #             adv_loss = self.adv_loss_fn(batch, batch_y)
    #             style_loss, style_score, content_score, tv_score = self.compute_sty_loss(self.patch.optmized_patch,  self.patch.init_patch, self.args['style_weight'], self.args['content_weight'], self.args['tv_weight'])
                
    #             loss = self.args['lambda']*style_loss + adv_loss
    #             if self.args['update']=='lbfgs':
    #                 loss.backward()

    #             if self.args['train_quan_patch_flag'] and (epoch+1)%10==0:
    #                 optmized_patch = (self.patch.optmized_patch*255).int().float()/255.
    #                 self.patch.optmized_patch.data = optmized_patch.data
                
    #             if self.args['train_log_flag']:
    #                 if epoch % self.args['train_img_log_interval']==0 or (epoch+1)==self.args['epoch']:
    #                     # ==== [추가 시작]: SSIM 계산 ====
    #                     # PyTorch 텐서(1, 3, H, W)를 Numpy 배열(H, W, 3)로 변환
    #                     orig_img_np = self.patch.init_patch[0].detach().cpu().numpy().transpose(1, 2, 0)
    #                     opt_img_np = self.patch.optmized_patch[0].detach().cpu().numpy().transpose(1, 2, 0)
                        
    #                     # SSIM 계산 (data_range는 0~1 이미지이므로 1.0으로 설정)
    #                     ssim_val = compare_ssim(orig_img_np, opt_img_np, multichannel=True, channel_axis=2, data_range=1.0)
                        
    #                     # 텐서보드에 SSIM 기록
    #                     self.log.add_scalar(f'{name_prefix}/train/ssim', ssim_val, epoch)
    #                     # 콘솔에도 출력
    #                     tqdm.write(f"[Epoch {epoch+1}] SSIM: {ssim_val:.4f} (1.0에 가까울수록 원본과 유사)")
    #                     # ==== [추가 끝] ====
                        
    #                     log_img_train(self.log, epoch, name_prefix, [adv_scene_image, ben_scene_image, (self.patch.optmized_patch*255).int().float()/255., batch_y[0], batch_y[1]])
    #                     self.log.add_image(f'{name_prefix}/train/object_mask', object_full_mask.detach().cpu()[0, 0], epoch, dataformats='HW')
                        
    #                     log_scale_train(self.log,epoch, name_prefix, style_score, content_score, tv_score, adv_loss, mean_shift, max_shift, min_shift, mean_ori, arr)
                
    #             if self.args['inner_eval_flag']:
    #                 if epoch % self.args['inner_eval_interval']==0 or (epoch+1)==self.args['epoch']: 
    #                     for category in self.objects.object_imgs_test.keys():
    #                         if self.args['random_test_flag']:
    #                             record = [[] for i in range(5)]
    #                             for _ in range(20):
    #                                 record_tmp = self.eval(self.MDE, category)
    #                                 for i in range(5):
    #                                     record[i]+=record_tmp[i]       
    #                         else:
    #                             record = self.eval(self.MDE, category)
                            
    #                         # ✨ [추가할 부분 시작] ✨
    #                         # 중간 평가된 Eval MRSR(record[3]의 평균)을 터미널에 출력해 줍니다.
    #                         current_eval_mrsr = np.mean(record[3])
    #                         tqdm.write(f"   🔍 [Eval 중간점검] Epoch {epoch+1} | 실전 Test MRSR: {current_eval_mrsr:.4f}")
    #                         # ✨ [추가할 부분 끝] ✨
                            
                            
    #                         log_scale_eval(self.log, epoch, name_prefix, self.MDE[0],category, np.mean(record[0]), np.mean(record[1]), np.mean(record[2]),np.mean(record[3]),np.mean(record[4]) )      
    #             return loss
            
    #         # --- [수정 및 추가 시작]: 디버깅 출력 (Loss 확인) ---
    #         if self.args['update']=='lbfgs':
    #             optimizer.zero_grad()
    #             loss = optimizer.step(closure) # LBFGS는 step()의 반환값으로 loss를 줍니다.
    #             LR_decay.step()
    #         else:
    #             loss = closure()
    #             grad = torch.autograd.grad(loss, [self.patch.optmized_patch] )[0]
                
    #             # 🌟 [추가] BIM을 위한 Learning Rate Decay (에폭이 지날수록 세밀하게 깎음)
    #             # 초기 LR에서 시작해서, 마지막 에폭에는 초기 LR의 1/10 수준으로 줄어들게 합니다.
    #             # current_lr = self.args['learning_rate'] * (1 - 0.9 * (epoch / self.args['epoch']))
                
    #             self.patch.optmized_patch = bim(grad, self.patch.optmized_patch,self.args['learning_rate'])
    #             # 수정된 current_lr을 bim 함수에 전달
    #             # self.patch.optmized_patch = bim(grad, self.patch.optmized_patch, current_lr)

    #         # --- [수정 및 추가]: 현재 Loss 출력 및 Best Patch 갱신 ---
    #         current_loss = loss.item()
    #         if (epoch + 1) % 10 == 0:
    #             # 👈 [수정] Loss와 함께 Train MRSR도 출력되게 변경
    #             current_mrsr = getattr(self, 'current_train_mrsr', 0.0)
    #             tqdm.write(f"👉 [Epoch {epoch+1}/{self.args['epoch']}] Total Loss: {current_loss:.4f} | Train MRSR: {current_mrsr:.4f}")
                
    #         # 🚨 [핵심 수정]: 초반 100 에폭(워밍업) 동안의 가짜 마이너스 Loss는 무시합니다! 🚨
    #         # 에폭이 100 이상 진행되어 Loss가 안정화된 이후부터 진짜 Best를 찾습니다.
    #         if epoch >= 100 and current_loss < best_loss:
    #             best_loss = current_loss
    #             best_patch = self.patch.optmized_patch.clone().detach()
    #             best_epoch = epoch + 1  # 👈 [추가] Best 갱신 시 에폭 번호 저장
                
    #         # (안전장치) 아직 100 에폭이 안 지나서 best_patch가 None일 경우를 대비해 초기화
    #         if best_patch is None:
    #             best_patch = self.patch.optmized_patch.clone().detach()
    #             best_epoch = epoch + 1  # 👈 [추가]
    #         # --------------------------------------------------------
            
    #     # --- [수정]: for 루프 종료 직후, 슬래시 에러 방지 및 Best 이미지 저장 ---
    #     # 1. 파일 이름에 들어간 '/'나 '\'를 '_'로 변경하여 폴더 생성 에러 방지
    #     safe_name_prefix = name_prefix.replace('/', '_').replace('\\', '_')
        
    #     save_dir = os.path.join(self.args.get('log_dir', 'runs'), "final_patches")
    #     os.makedirs(save_dir, exist_ok=True)
    
    #     # 2. 마지막 시점이 아닌, 가장 Loss가 낮았던 'best_patch'를 저장
    #     save_path = os.path.join(save_dir, f"{safe_name_prefix}_best.png")
    #     final_patch_img = torch.clamp(best_patch, 0, 1).cpu()
    #     save_image(final_patch_img, save_path)
        
    #     # 1. Best Patch의 SSIM 계산 (은밀성 평가)
    #     # PyTorch 텐서(1, 3, H, W) -> Numpy 배열(H, W, 3) 변환
    #     orig_img_np = self.patch.init_patch[0].detach().cpu().numpy().transpose(1, 2, 0)
    #     best_img_np = final_patch_img[0].detach().cpu().numpy().transpose(1, 2, 0)
        
    #     best_ssim = compare_ssim(orig_img_np, best_img_np, multichannel=True, channel_axis=2, data_range=1.0)
        
    #     # 2. Best Patch의 Eval MRSR, ARR 계산 (공격력 평가)
    #     mrsr_list = []
    #     arr_list = []
    #     if self.args.get('inner_eval_flag', False):
    #         # 모든 테스트 카테고리(car 등)에 대해 평가 수행
    #         for category in self.objects.object_imgs_test.keys():
    #             # ✨ 핵심: 현재 훈련 중인 패치가 아닌 'best_patch'를 인자로 넘겨서 평가합니다!
    #             record = self.eval(self.MDE, category, patch=best_patch)
    #             mrsr_list.extend(record[3]) # record[3]가 MRSR
    #             arr_list.extend(record[4])  # record[4]가 ARR
        
    #     final_mrsr = np.mean(mrsr_list) if mrsr_list else 0.0
    #     final_arr = np.mean(arr_list) if arr_list else 0.0

    #     # 3. 결과 콘솔 출력
    #     tqdm.write("\n" + "="*60)
    #     tqdm.write("✅ 최적화 완료! 가장 Loss가 낮았던 최적 패치 저장됨")
    #     tqdm.write(f"📂 경로: {save_path}")
    #     tqdm.write("-" * 60)
    #     tqdm.write("📊 [최적 패치(Best Patch) 최종 성능 요약]")
    #     tqdm.write(f"   🏆 Best Epoch   : {best_epoch} 번째 에폭")  # 👈 [추가] 언제 베스트였는지 출력
    #     tqdm.write(f"   📉 Total Loss   : {best_loss:.4f}")
    #     tqdm.write(f"   🎨 SSIM (유사도): {best_ssim:.4f} (1.0에 가까울수록 원본과 유사/은밀함)")
    #     if self.args.get('inner_eval_flag', False):
    #         tqdm.write(f"   🎯 Eval MRSR  : {final_mrsr:.4f} (높을수록 모델을 강하게 속임)")
    #         tqdm.write(f"   🎯 Eval ARR   : {final_arr:.4f} (높을수록 넓은 영역을 속임)")
    #     tqdm.write("="*60 + "\n")
        
        # # --- [수정 끝] ---
        # return best_loss, best_ssim, final_mrsr # <--- [추가] 튜닝을 위해 최종 Best Loss를 반환
    def run_with_random_object(self, scene_dir, scene_file, idx, points):
        self.env = ENV(self.args, scene_file, scene_dir, idx, points)
        name_prefix = f"{self.args['patch_file'][:-4]}_{scene_file[:-4]}_{idx}"
        if self.args['update']=='lbfgs':
            optimizer = optim.LBFGS([self.patch.optmized_patch], lr=self.args['learning_rate'])
            LR_decay = PolynomialLRDecay(optimizer, self.args['epoch']//2, self.args['learning_rate']/2, 0.9)
            
        # 🌟 [수정]: 스코프(Scope) 에러 방지 및 추적을 위해 self 변수로 초기화
        self.best_loss = float('inf')
        self.best_loss_epoch = 0      # Total Loss가 가장 낮았던 에폭 추적용
        
        self.best_eval_mrsr = -float('inf')
        self.best_patch = None
        self.best_epoch = 0           # Test MRSR이 가장 높았던 에폭 추적용

        for epoch in tqdm(range(self.args['epoch']), desc=f"Training {idx}/{self.args['scene_num']}"):
            def closure(): 
                self.patch.optmized_patch.data.clamp_(0, 1)
                batch, patch_size = self.env.accept_patch_and_objects(False, self.patch.optmized_patch, self.patch.mask, self.objects.object_imgs_train, self.env.insert_range, None, None, offset_patch=self.args['train_offset_patch_flag'], color_patch=self.args['train_color_patch_flag'], offset_object=self.args['train_offset_object_flag'], color_object=self.args['train_color_object_flag'])

                adv_scene_image, ben_scene_image, scene_img, patch_full_mask, object_full_mask = batch
                
                # ==== 패치 & 장애물 삽입 확인용 이미지 저장 ====
                if epoch == 0 or (epoch + 1) % 100 == 0:
                    safe_name_prefix = name_prefix.replace('/', '_').replace('\\', '_')
                    debug_dir = os.path.join(self.args.get('log_dir', 'runs'), "debug_scene")
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    save_image(adv_scene_image.clone().detach().cpu(), 
                               os.path.join(debug_dir, f"{safe_name_prefix}_ep{epoch+1}_scene.png"))
                    save_image(patch_full_mask.clone().detach().cpu(), 
                               os.path.join(debug_dir, f"{safe_name_prefix}_ep{epoch+1}_patch_mask.png"))
                    save_image(object_full_mask.clone().detach().cpu(), 
                               os.path.join(debug_dir, f"{safe_name_prefix}_ep{epoch+1}_object_mask.png"))
                
                batch= [ torch.nn.functional.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch]
                batch_y = predict_batch(batch, self.MDE)
                
                mean_ori, mean_shift, max_shift, min_shift, arr =self.eval_core(batch_y[0], batch_y[1], batch[-1])
                
                # 👈 [추가] 현재 10단위 출력을 위해 현재 배치의 Train MRSR을 계산해둡니다.
                self.current_train_mrsr = (mean_shift / mean_ori).item() if mean_ori.item() != 0 else 0.0
                
                adv_loss = self.adv_loss_fn(batch, batch_y)
                style_loss, style_score, content_score, tv_score = self.compute_sty_loss(self.patch.optmized_patch,  self.patch.init_patch, self.args['style_weight'], self.args['content_weight'], self.args['tv_weight'])
                
                loss = self.args['lambda']*style_loss + adv_loss
                if self.args['update']=='lbfgs':
                    loss.backward()

                if self.args['train_quan_patch_flag'] and (epoch+1)%10==0:
                    optmized_patch = (self.patch.optmized_patch*255).int().float()/255.
                    self.patch.optmized_patch.data = optmized_patch.data
                
                if self.args['train_log_flag']:
                    if epoch % self.args['train_img_log_interval']==0 or (epoch+1)==self.args['epoch']:
                        # SSIM 계산
                        orig_img_np = self.patch.init_patch[0].detach().cpu().numpy().transpose(1, 2, 0)
                        opt_img_np = self.patch.optmized_patch[0].detach().cpu().numpy().transpose(1, 2, 0)
                        ssim_val = compare_ssim(orig_img_np, opt_img_np, multichannel=True, channel_axis=2, data_range=1.0)
                        
                        self.log.add_scalar(f'{name_prefix}/train/ssim', ssim_val, epoch)
                        tqdm.write(f"[Epoch {epoch+1}] SSIM: {ssim_val:.4f} (1.0에 가까울수록 원본과 유사)")
                        
                        log_img_train(self.log, epoch, name_prefix, [adv_scene_image, ben_scene_image, (self.patch.optmized_patch*255).int().float()/255., batch_y[0], batch_y[1]])
                        self.log.add_image(f'{name_prefix}/train/object_mask', object_full_mask.detach().cpu()[0, 0], epoch, dataformats='HW')
                        log_scale_train(self.log,epoch, name_prefix, style_score, content_score, tv_score, adv_loss, mean_shift, max_shift, min_shift, mean_ori, arr)
                
                # 🌟 [수정된 중간 평가 및 Early Stopping 로직] 🌟
                if self.args['inner_eval_flag']:
                    interval = self.args.get('inner_eval_interval', 100)
                    if epoch % interval == 0 or (epoch+1) == self.args['epoch']: 
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
                        
                        # ✨ 현재 Test MRSR 평균값 계산 및 self 변수로 갱신
                        current_eval_mrsr = np.mean(record[3])
                        
                        if current_eval_mrsr > self.best_eval_mrsr:
                            self.best_eval_mrsr = current_eval_mrsr
                            self.best_patch = self.patch.optmized_patch.clone().detach()
                            self.best_epoch = epoch + 1
                            tqdm.write(f"   🔥 [New Best!] Epoch {epoch+1} | 실전 Test MRSR: {current_eval_mrsr:.4f} (갱신됨!)")
                        else:
                            tqdm.write(f"   🔍 [중간점검] Epoch {epoch+1} | 실전 Test MRSR: {current_eval_mrsr:.4f}")
                return loss
            
            if self.args['update']=='lbfgs':
                optimizer.zero_grad()
                loss = optimizer.step(closure) 
                LR_decay.step()
            else:
                loss = closure()
                grad = torch.autograd.grad(loss, [self.patch.optmized_patch] )[0]
                
                # 🌟 [추가] BIM Learning Rate Decay
                # current_lr = self.args['learning_rate'] * (1 - 0.9 * (epoch / self.args['epoch']))
                self.patch.optmized_patch = bim(grad, self.patch.optmized_patch, self.args['learning_rate'])

            # --- [수정 및 추가]: 현재 Loss 및 Train MRSR 출력 ---
            current_loss = loss.item()
            
            # 👈 [추가]: Total Loss 최저점 추적
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_loss_epoch = epoch + 1
                
            if (epoch + 1) % 10 == 0:
                current_mrsr = getattr(self, 'current_train_mrsr', 0.0)
                tqdm.write(f"👉 [Epoch {epoch+1}/{self.args['epoch']}] Total Loss: {current_loss:.4f} | Train MRSR: {current_mrsr:.4f}")
                
            # 안전장치
            if self.best_patch is None:
                self.best_patch = self.patch.optmized_patch.clone().detach()
                self.best_epoch = epoch + 1
            
        safe_name_prefix = name_prefix.replace('/', '_').replace('\\', '_')
        save_dir = os.path.join(self.args.get('log_dir', 'runs'), "final_patches")
        os.makedirs(save_dir, exist_ok=True)
    
        save_path = os.path.join(save_dir, f"{safe_name_prefix}_best.png")
        final_patch_img = torch.clamp(self.best_patch, 0, 1).cpu()
        save_image(final_patch_img, save_path)
        
        orig_img_np = self.patch.init_patch[0].detach().cpu().numpy().transpose(1, 2, 0)
        best_img_np = final_patch_img[0].detach().cpu().numpy().transpose(1, 2, 0)
        best_ssim = compare_ssim(orig_img_np, best_img_np, multichannel=True, channel_axis=2, data_range=1.0)
        
        mrsr_list = []
        arr_list = []
        if self.args.get('inner_eval_flag', False):
            for category in self.objects.object_imgs_test.keys():
                record = self.eval(self.MDE, category, patch=self.best_patch)
                mrsr_list.extend(record[3]) 
                arr_list.extend(record[4])  
        
        final_mrsr = np.mean(mrsr_list) if mrsr_list else 0.0
        final_arr = np.mean(arr_list) if arr_list else 0.0

        tqdm.write("\n" + "="*60)
        tqdm.write("✅ 최적화 완료! 최고의 Test 성능을 낸 패치 저장됨")
        tqdm.write(f"📂 경로: {save_path}")
        tqdm.write("-" * 60)
        tqdm.write("📊 [최적 패치(Best Patch) 최종 성능 요약]")
        tqdm.write(f"   🏆 Best Eval MRSR Epoch : {self.best_epoch} 번째 에폭")
        tqdm.write(f"   📉 Best Total Loss Epoch: {self.best_loss_epoch} 번째 에폭 (Loss: {self.best_loss:.4f})")
        tqdm.write(f"   🎨 SSIM (유사도): {best_ssim:.4f} (1.0에 가까울수록 원본과 유사/은밀함)")
        if self.args.get('inner_eval_flag', False):
            tqdm.write(f"   🎯 Eval MRSR  : {final_mrsr:.4f} (높을수록 모델을 강하게 속임)")
            tqdm.write(f"   🎯 Eval ARR   : {final_arr:.4f} (높을수록 넓은 영역을 속임)")
        tqdm.write("="*60 + "\n")
        
        return self.best_loss, best_ssim, final_mrsr
 
    def run_with_fixed_object(self, scene_dir, scene_file, idx, points):
        self.env = ENV(self.args, scene_file, scene_dir, idx, points)
        name_prefix = f"{self.args['patch_file'][:-4]}_{scene_file[:-4]}_{idx}"
        if self.args['update']=='lbfgs':
            optimizer = optim.LBFGS([self.patch.optmized_patch], lr=self.args['learning_rate'])
            LR_decay = PolynomialLRDecay(optimizer, self.args['epoch']//2, self.args['learning_rate']/2, 0.9)

        for epoch in tqdm(range(self.args['epoch']), desc=f"Training {idx}/{self.args['scene_num']}"):
            def closure(): 
                if self.args['update']=='lbfgs':
                    optimizer.zero_grad()
                self.patch.optmized_patch.data.clamp_(0, 1)
                adv_scene_image, ben_scene_image, patch_size, patch_full_mask = self.env.accept_patch(self.patch.optmized_patch, None, self.patch.mask, self.args['insert_height'], offset_patch=self.args['train_offset_patch_flag'], color_patch=self.args['train_color_patch_flag'])

                batch= [adv_scene_image, ben_scene_image, self.env.env, patch_full_mask, self.objects.object_full_mask]
                batch= [ torch.nn.functional.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch]

                batch_y = predict_batch(batch, self.MDE)
                
                mean_ori, mean_shift, max_shift, min_shift, arr =self.eval_core(batch_y[0], batch_y[1], batch[-1])
                
                adv_loss = self.adv_loss_fn(batch, batch_y)
                style_loss, style_score, content_score, tv_score = self.compute_sty_loss(self.patch.optmized_patch,  self.patch.init_patch, self.args['style_weight'], self.args['content_weight'], self.args['tv_weight'])
                
                loss = self.args['lambda']*style_loss + adv_loss
                if self.args['update']=='lbfgs':
                    loss.backward()

                if self.args['train_quan_patch_flag'] and (epoch+1)%10==0:
                    optmized_patch = (self.patch.optmized_patch*255).int().float()/255.
                    self.patch.optmized_patch.data = optmized_patch.data
                if self.args['train_log_flag']:
                    if epoch % self.args['train_img_log_interval']==0 or (epoch+1)==self.args['epoch']:
                        log_img_train(self.log, epoch, name_prefix, [adv_scene_image, ben_scene_image, (self.patch.optmized_patch*255).int().float()/255., batch_y[0], batch_y[1]])
                        log_scale_train(self.log,epoch, name_prefix, style_score, content_score, tv_score, adv_loss, mean_shift, max_shift, min_shift, mean_ori, arr)
                
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
        if category == 'pas':
             object_num=3
        else:
            object_num=1

        with torch.no_grad():
            record = [[] for _ in range(5)]
            for idx in range(len(self.objects.object_imgs_test[category])-object_num+1):
                batch, _ = self.env.accept_patch_and_objects(True, patch, self.patch.mask, self.objects.object_imgs_test, self.env.insert_range, insert_height, None, offset_patch=self.args['test_offset_patch_flag'], color_patch=self.args['test_color_patch_flag'], offset_object=self.args['test_offset_object_flag'], color_object=self.args['test_color_object_flag'],object_idx_g=idx, category=category)
                
                batch= [ torch.nn.functional.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch]
                batch_y= predict_batch(batch, MDE)
                
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


    # def run_with_fixed_object_multi_frame(self, scene_dir, scene_file, idx, points):
        
    #     csv_file = self.args['csv_dir']
    #     scene_set =  pd.read_csv(csv_file)
    #     self.env1 = ENV(self.args, 'phy_ntu_1.png', scene_dir, 0, points = scene_set.iloc[0])
    #     self.env2 = ENV(self.args, 'phy_ntu_2.png', scene_dir, 1, points = scene_set.iloc[1])
    #     self.env3 = ENV(self.args, 'phy_ntu_3.png', scene_dir, 2, points = scene_set.iloc[2])
    #     self.env4 = ENV(self.args, 'phy_ntu_4.png', scene_dir, 3, points = scene_set.iloc[3])

    #     self.objects1 = OBJ(self.args)
    #     self.objects1.load_object_mask('phy_ntu_1_mask.png', self.args['obj_full_mask_dir'])
    #     self.objects2 = OBJ(self.args)
    #     self.objects2.load_object_mask('phy_ntu_2_mask.png', self.args['obj_full_mask_dir'])
    #     self.objects3 = OBJ(self.args)
    #     self.objects3.load_object_mask('phy_ntu_3_mask.png', self.args['obj_full_mask_dir'])
    #     self.objects4 = OBJ(self.args)
    #     self.objects4.load_object_mask('phy_ntu_4_mask.png', self.args['obj_full_mask_dir'])
        
    #     name_prefix1 = f"{self.args['patch_file'][:-4]}_phy_ntu_1_{idx}"
    #     name_prefix2 = f"{self.args['patch_file'][:-4]}_phy_ntu_2_{idx}"
    #     name_prefix3 = f"{self.args['patch_file'][:-4]}_phy_ntu_3_{idx}"
    #     name_prefix4 = f"{self.args['patch_file'][:-4]}_phy_ntu_4_{idx}"
    #     if self.args['update']=='lbfgs':
    #         optimizer = optim.LBFGS([self.patch.optmized_patch], lr=self.args['learning_rate'])
    #         LR_decay = PolynomialLRDecay(optimizer, self.args['epoch']//2, self.args['learning_rate']/2, 0.9)

    #     for epoch in tqdm(range(self.args['epoch']), desc=f"Training {idx}/{self.args['scene_num']}"):
    #         def closure(): 
    #             if self.args['update']=='lbfgs':
    #                 optimizer.zero_grad()
    #             self.patch.optmized_patch.data.clamp_(0, 1)
                
    #             bri,con,sat=random.uniform(0.8,1.2),random.uniform(0.9,1.1),random.uniform(0.9,1.1)
    #             adv_scene_image1, ben_scene_image1, patch_size1, patch_full_mask1 = self.env1.accept_patch(self.patch.optmized_patch, None, self.patch.mask, 496, offset_patch=self.args['train_offset_patch_flag'], color_patch=self.args['train_color_patch_flag'],patch_heigh=298,brightness=bri,contrast=con,saturation=sat)
    #             adv_scene_image2, ben_scene_image2, patch_size2, patch_full_mask2 = self.env2.accept_patch(self.patch.optmized_patch, None, self.patch.mask, 446, offset_patch=self.args['train_offset_patch_flag'], color_patch=self.args['train_color_patch_flag'],patch_heigh=219,brightness=bri,contrast=con,saturation=sat)
    #             adv_scene_image3, ben_scene_image3, patch_size3, patch_full_mask3 = self.env3.accept_patch(self.patch.optmized_patch, None, self.patch.mask, 406, offset_patch=self.args['train_offset_patch_flag'], color_patch=self.args['train_color_patch_flag'],patch_heigh=166,brightness=bri,contrast=con,saturation=sat)
    #             adv_scene_image4, ben_scene_image4, patch_size4, patch_full_mask4 = self.env4.accept_patch(self.patch.optmized_patch, None, self.patch.mask, 368, offset_patch=self.args['train_offset_patch_flag'], color_patch=self.args['train_color_patch_flag'],patch_heigh=134,brightness=bri,contrast=con,saturation=sat)

    #             batch1= [adv_scene_image1, ben_scene_image1, self.env1.env, patch_full_mask1, self.objects1.object_full_mask]
    #             batch2= [adv_scene_image2, ben_scene_image2, self.env2.env, patch_full_mask2, self.objects2.object_full_mask]
    #             batch3= [adv_scene_image3, ben_scene_image3, self.env3.env, patch_full_mask3, self.objects3.object_full_mask]
    #             batch4= [adv_scene_image4, ben_scene_image4, self.env4.env, patch_full_mask4, self.objects4.object_full_mask]
    #             batch1= [ torch.nn.functional.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch1]
    #             batch2= [ torch.nn.functional.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch2]
    #             batch3= [ torch.nn.functional.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch3]
    #             batch4= [ torch.nn.functional.interpolate(item,size=([int(self.args['input_height']),int(self.args['input_width'])])) for item in batch4]

    #             batch_y1 = predict_batch(batch1, self.MDE)
    #             batch_y2 = predict_batch(batch2, self.MDE)
    #             batch_y3 = predict_batch(batch3, self.MDE)
    #             batch_y4 = predict_batch(batch4, self.MDE)
                
    #             adv_loss1 = self.adv_loss_fn(batch1, batch_y1)
    #             adv_loss2 = self.adv_loss_fn(batch2, batch_y2)
    #             adv_loss3 = self.adv_loss_fn(batch3, batch_y3)
    #             adv_loss4 = self.adv_loss_fn(batch4, batch_y4)
    #             adv_loss = (adv_loss1+adv_loss2+adv_loss3+adv_loss4)/4

    #             mean_ori1, mean_shift1, max_shift1, min_shift1, arr1 =self.eval_core(batch_y1[0], batch_y1[1], batch1[-1])
    #             mean_ori2, mean_shift2, max_shift2, min_shift2, arr2 =self.eval_core(batch_y2[0], batch_y2[1], batch2[-1])
    #             mean_ori3, mean_shift3, max_shift3, min_shift3, arr3 =self.eval_core(batch_y3[0], batch_y3[1], batch3[-1])
    #             mean_ori4, mean_shift4, max_shift4, min_shift4, arr4 =self.eval_core(batch_y4[0], batch_y4[1], batch4[-1])
                
    #             # adv_loss = self.adv_loss_fn(batch, batch_y)
    #             style_loss, style_score, content_score, tv_score = self.compute_sty_loss(self.patch.optmized_patch,  self.patch.init_patch, self.args['style_weight'], self.args['content_weight'], self.args['tv_weight'])
                
    #             loss = self.args['lambda']*style_loss + self.args['beta']*adv_loss
    #             if self.args['update']=='lbfgs':
    #                 loss.backward()

    #             if self.args['train_quan_patch_flag'] and (epoch+1)%10==0:
    #                 optmized_patch = (self.patch.optmized_patch*255).int().float()/255.
    #                 self.patch.optmized_patch.data = optmized_patch.data

    #             if epoch % self.args['train_img_log_interval']==0 or (epoch+1)==self.args['epoch']:
                    
    #                 log_img_train(self.log, epoch, name_prefix1, [adv_scene_image1, ben_scene_image1, (self.patch.optmized_patch*255).int().float()/255., batch_y1[0], batch_y1[1]])

    #                 log_img_train(self.log, epoch, name_prefix2, [adv_scene_image2, ben_scene_image2, (self.patch.optmized_patch*255).int().float()/255., batch_y2[0], batch_y2[1]])

    #                 log_img_train(self.log, epoch, name_prefix3, [adv_scene_image3, ben_scene_image3, (self.patch.optmized_patch*255).int().float()/255., batch_y3[0], batch_y3[1]])

    #                 log_img_train(self.log, epoch, name_prefix4, [adv_scene_image4, ben_scene_image4, (self.patch.optmized_patch*255).int().float()/255., batch_y4[0], batch_y4[1]])

    #                 log_scale_train(self.log,epoch, name_prefix1, style_score, content_score, tv_score, adv_loss, mean_shift1, max_shift1, min_shift1, mean_ori1, arr1)
    #                 log_scale_train(self.log,epoch, name_prefix2, style_score, content_score, tv_score, adv_loss, mean_shift2, max_shift2, min_shift2, mean_ori2, arr1)
    #                 log_scale_train(self.log,epoch, name_prefix3, style_score, content_score, tv_score, adv_loss, mean_shift3, max_shift3, min_shift3, mean_ori3, arr3)
    #                 log_scale_train(self.log,epoch, name_prefix4, style_score, content_score, tv_score, adv_loss, mean_shift4, max_shift4, min_shift4, mean_ori4, arr4)
                
    #             return loss
    #         if self.args['update']=='lbfgs':

    #             optimizer.step(closure)
    #             LR_decay.step()
    #         else:
    #             loss = closure()
    #             grad = torch.autograd.grad(loss, [self.patch.optmized_patch] )[0]
    #             self.patch.optmized_patch = bim(grad, self.patch.optmized_patch,self.args['learning_rate'])
    



    