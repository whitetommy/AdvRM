from option import get_args
from advrm import ADVRM
import pandas as pd
import time
from log import record_hparams, log_scale_eval
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import random
import os, gc

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run_single_scene(args, scene_set, idx, patch_size, base_log_dir):
    """단일 씬에 대한 최적화를 수행하고 텐서보드 및 로컬 폴더를 분리하는 함수"""
    # 🚨 팩트 보호: idx가 데이터 길이보다 크면 실행을 건너뜀 (안전장치 1)
    if idx >= len(scene_set):
        print(f"⚠️ Warning: Index {idx} exceeds the number of available scenes ({len(scene_set)}). Skipping.")
        return

    scene_file = scene_set['Filename'].to_list()[idx]
    scene_key_points = scene_set.iloc[idx]
    scene_name = scene_file[:-4]

    print(f"\n{'='*60}")
    print(f"🚀 Start Optimizing Patch for Scene {idx:03d} : {scene_file}")
    print(f"{'='*60}")

    scene_log_dir = os.path.join(base_log_dir, f"scene_{idx:03d}_{scene_name}")
    writer = SummaryWriter(log_dir=scene_log_dir, comment=args.get('log_dir_comment', ''))

    args['current_scene_idx'] = idx
    args['current_scene_name'] = scene_name

    if not args['random_object_flag']:
        args['obj_full_mask_file'] = f'{scene_name}_mask.png'
        
    advrm = ADVRM(args, writer, patch_size, True, args['random_object_flag'])

    final_e_blend, final_e_cover, final_ssim = advrm.run(args['scene_dir'], scene_file, idx, scene_key_points)
    
    print(f"✅ [Scene {idx:03d} Result] E_BLEND: {final_e_blend:.4f} | E_COVER: {final_e_cover:.4f} | SSIM: {final_ssim:.4f}")

    writer.close()

    # === 🚨 VRAM 누수 방지 코어 (핵심 부분) ===
    # 파이토치의 모델과 텐서보드 Writer 등이 잡고있는 캐시를 강제로 파괴
    if hasattr(advrm, 'MDE'): advrm.MDE = None
    if hasattr(advrm, 'style_model'): advrm.style_model = None
    if hasattr(advrm, 'objects'): advrm.objects = None
    if hasattr(advrm, 'env'): advrm.env = None
    advrm.style_losses = []
    advrm.content_losses = []
    
    del advrm
    gc.collect()
    torch.cuda.empty_cache()
    # ============================================

def main(args):
    current = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    
    comment = args.get('log_dir_comment', 'multi_scene').strip()
    base_log_dir = os.path.join(args['log_dir'], f"{current}_{comment}")
    # base_log_dir = os.path.join(args['log_dir'], "2026-04-07-18-57-51_")
    os.makedirs(base_log_dir, exist_ok=True)

    csv_file = args['csv_dir']
    scene_set = pd.read_csv(csv_file)

    if len(args['patch_size'].split(',')) == 2:
        patch_size = [int(i) for i in args['patch_size'].split(',')]
    else:
        patch_size = None
    
    idx = args['idx']
    if idx >= 0:
        run_single_scene(args, scene_set, idx, patch_size, base_log_dir)
    else:
        # 🚨 팩트 보호: args['scene_num']과 CSV 데이터 길이 중 작은 값까지만 반복 (안전장치 2)
        max_scenes = min(args['scene_num'], len(scene_set))

        # 🚨 028번 씬부터 재개하기 위한 강제 세팅
        # start_idx = 28
        # print(f"📌 Total scenes to process: {max_scenes} (Max available: {len(scene_set)}) | Resuming from: {start_idx}")

        print(f"📌 Total scenes to process: {max_scenes} (Max available: {len(scene_set)})")
        

        for i in range(max_scenes):
        # for i in range(start_idx, max_scenes):
            run_single_scene(args, scene_set, i, patch_size, base_log_dir)

        summary_path = os.path.join(base_log_dir, "experiment_summary.csv")
        if os.path.exists(summary_path):
            df_all = pd.read_csv(summary_path)
            total_avg = pd.DataFrame([{
                'Scene_Idx': 'TOTAL_AVG',
                'Scene_Name': 'ALL_SCENES',
                'Avg_E_Blend': df_all['Avg_E_Blend'].mean(),
                'Avg_E_Cover': df_all['Avg_E_Cover'].mean(),
                'SSIM': '-',
                'Epoch': '-'
            }])
            total_avg.to_csv(summary_path, mode='a', header=False, index=False)
            print(f"🏁 실험 종료! 전체 평균이 {summary_path}에 추가되었습니다.")

if __name__=='__main__':
    args = get_args()
    torch.cuda.set_device(args['device'])
    setup_seed(args['seed'])
    main(args)