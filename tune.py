# import optuna
# import subprocess
# import numpy as np

# def objective(trial):
#     # 1. 탐색 공간 (Adam에 맞춰서 lr 하한선을 더 낮춤)
#     lr = trial.suggest_float("learning_rate", 0.0001, 0.01, log=True)
#     lam = trial.suggest_float("lambda", 0.000000001, 0.000001, log=True)
#     tv_weight = trial.suggest_float("tv_weight", 1e-8, 1e-3, log=True)
#     # update_method = trial.suggest_categorical("update", ["adam", "bim"])

#     epochs = "1000" 
#     target_scene = 0 # 🌟 단일 씬 (idx=0) 고정!
    
#     print(f"\n{'='*60}")
#     print(f"[{trial.number}번째 Trial 시작] lr={lr:.6f}, lambda={lam:.2e}, tv={tv_weight:.2e}, update={update_method}")
#     print(f"{'='*60}")

#     cmd = [
#         "python", "run.py",
#         "--depth_model", "monodepth2",
#         "--idx", str(target_scene), 
#         "--update", "adam",
#         "--patch_file", "6.jpg",
#         "--epoch", epochs,
#         "--lambda", str(lam),
#         "--patch_dir", "./asset/patch",
#         "--obj_dir", "./asset/obstacle",
#         "--learning_rate", str(lr),
#         "--tv_weight", str(tv_weight),
#         "--train_log_flag", "--random_object_flag", "--inner_eval_flag",
#         "--train_color_patch_flag", "--test_color_patch_flag",
#         "--train_color_object_flag", "--test_color_object_flag",
#         "--train_quan_patch_flag", "--test_quan_patch_flag",
#         "--random_test_flag"
#     ]
    
#     print(f"  ▶ Scene {target_scene} 훈련 시작...")
#     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
#     scene_e_blend, scene_ssim, scene_train_gap = None, None, None
    
#     # 2. 실시간 로그 파싱
#     for line in process.stdout:
#         # print(line, end="") # 에러 발생 시 주석 풀고 확인
#         if "OPTUNA_FINAL_E_BLEND:" in line:
#             scene_e_blend = float(line.strip().split(":")[-1])
#         elif "OPTUNA_FINAL_SSIM:" in line:
#             scene_ssim = float(line.strip().split(":")[-1])
#         elif "OPTUNA_FINAL_TRAIN_GAP:" in line:
#             scene_train_gap = float(line.strip().split(":")[-1])
            
#     process.wait()

#     # 실행 실패 또는 지표 누락 시 예외 처리
#     if scene_e_blend is None or scene_ssim is None or scene_train_gap is None:
#         raise optuna.exceptions.TrialPruned(f"Scene {target_scene} 실행 중 오류 발생 (지표 누락)")
        
#     print(f"  ✔ Scene {target_scene} 완료 - E_blend: {scene_e_blend:.4f}, SSIM: {scene_ssim:.4f}, Train Gap: {scene_train_gap:.4f}")

#     # =================================================================
#     # 🌟 3. 최종 Score(페널티) 계산 (단일 씬이므로 바로 계산) 🌟
#     # =================================================================
#     score = scene_e_blend  

#     # 제약 조건 1: 무늬 방어선 (SSIM < 0.40 이면 페널티)
#     ssim_threshold = 0.40
#     if scene_ssim < ssim_threshold:
#         score += (ssim_threshold - scene_ssim) * 10.0 # 가차 없이 10배 페널티
#     else:
#         score -= (scene_ssim - ssim_threshold) * 0.5  # 예쁘게 남으면 보너스

#     # 제약 조건 2: 훈련 수렴 방어선 (네가 설정한 빡센 0.6 기준!)
#     train_gap_threshold = 0.6
#     if scene_train_gap > train_gap_threshold:
#         print(f"  [경고] 훈련 수렴 실패! (Train Gap: {scene_train_gap:.4f}) -> 페널티 폭탄!")
#         score += (scene_train_gap - train_gap_threshold) * 5.0 

#     print(f"\n[{trial.number}번째 Trial 최종 결과] E_blend: {scene_e_blend:.4f}, SSIM: {scene_ssim:.4f}, Train Gap: {scene_train_gap:.4f}")
#     print(f"➔ 🏆 Final Score: {score:.4f}\n")

#     return score

# if __name__ == "__main__":
#     study = optuna.create_study(direction="minimize", study_name="AdvRM_Tuning_SingleScene")
    
#     try:
#         study.optimize(objective, n_trials=30)
#     except KeyboardInterrupt:
#         print("\n사용자에 의해 튜닝이 중단됨.")

#     print("\n" + "="*50)
#     print(f"🏆 Scene 0 전용 황금 베스트 하이퍼파라미터 🏆")
#     print(f"최소 Score: {study.best_value:.4f}")
#     for key, value in study.best_params.items():
#         print(f"  {key}: {value}")
#     print("="*50)

import optuna
import subprocess
import numpy as np

def objective(trial):
    # 1. 탐색 공간 (🌟 BIM 특성에 맞게 LR 범위 상향 조정)
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True) 
    lam = trial.suggest_float("lambda", 1e-8, 1e-3, log=True)
    tv_weight = trial.suggest_float("tv_weight", 1e-8, 1e-3, log=True)
    
    # 🌟 Adam 제거, 무조건 BIM으로 고정!
    update_method = "bim"

    epochs = "1000" 
    target_scene = 0 # 🌟 단일 씬 (idx=0) 고정!
    
    print(f"\n{'='*60}")
    print(f"[{trial.number}번째 Trial 시작] lr={lr:.6f}, lambda={lam:.2e}, tv={tv_weight:.2e}, update={update_method}")
    print(f"{'='*60}")

    cmd = [
        "python", "run.py",
        "--depth_model", "monodepth2",
        "--idx", str(target_scene), 
        "--update", update_method,
        "--patch_file", "6.jpg",
        "--epoch", epochs,
        "--lambda", str(lam),
        "--patch_dir", "./asset/patch",
        "--obj_dir", "./asset/obstacle",
        "--learning_rate", str(lr),
        "--tv_weight", str(tv_weight),
        "--train_log_flag", "--random_object_flag", "--inner_eval_flag",
        "--train_color_patch_flag", "--test_color_patch_flag",
        "--train_color_object_flag", "--test_color_object_flag",
        "--train_quan_patch_flag", "--test_quan_patch_flag",
        "--train_offset_patch_flag", "--test_offset_patch_flag",
        "--train_offset_object_flag", "--test_offset_object_flag",
        "--random_test_flag"
    ]
    
    print(f"  ▶ Scene {target_scene} 훈련 시작...")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    scene_e_blend, scene_ssim, scene_train_gap = None, None, None
    
    # 2. 실시간 로그 파싱
    for line in process.stdout:
        print(line, end="") # 에러 발생 시 주석 풀고 확인
        if "OPTUNA_FINAL_E_BLEND:" in line:
            scene_e_blend = float(line.strip().split(":")[-1])
        elif "OPTUNA_FINAL_SSIM:" in line:
            scene_ssim = float(line.strip().split(":")[-1])
        elif "OPTUNA_FINAL_TRAIN_GAP:" in line:
            scene_train_gap = float(line.strip().split(":")[-1])
            
    process.wait()

    # 실행 실패 또는 지표 누락 시 예외 처리
    if scene_e_blend is None or scene_ssim is None or scene_train_gap is None:
        raise optuna.exceptions.TrialPruned(f"Scene {target_scene} 실행 중 오류 발생 (지표 누락)")
        
    print(f"  ✔ Scene {target_scene} 완료 - E_blend: {scene_e_blend:.4f}, SSIM: {scene_ssim:.4f}, Train Gap: {scene_train_gap:.4f}")

    # =================================================================
    # 🌟 3. 최종 Score(페널티) 계산 (단일 씬이므로 바로 계산) 🌟
    # =================================================================
    score = scene_e_blend  

    # 제약 조건 1: 무늬 방어선 (SSIM < 0.40 이면 페널티)
    ssim_threshold = 0.3
    if scene_ssim < ssim_threshold:
        score += (ssim_threshold - scene_ssim) * 10.0 # 가차 없이 10배 페널티
    else:
        score -= (scene_ssim - ssim_threshold) * 0.5  # 예쁘게 남으면 보너스

    # 제약 조건 2: 훈련 수렴 방어선 (네가 설정한 빡센 0.6 기준!)
    train_gap_threshold = 0.6
    if scene_train_gap > train_gap_threshold:
        print(f"  [경고] 훈련 수렴 실패! (Train Gap: {scene_train_gap:.4f}) -> 페널티 폭탄!")
        score += (scene_train_gap - train_gap_threshold) * 5.0 

    print(f"\n[{trial.number}번째 Trial 최종 결과] E_blend: {scene_e_blend:.4f}, SSIM: {scene_ssim:.4f}, Train Gap: {scene_train_gap:.4f}")
    print(f"➔ 🏆 Final Score: {score:.4f}\n")

    return score

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", study_name="AdvRM_Tuning_SingleScene")
    
    try:
        # 단일 씬이라 빠르므로 30~50번 정도 시도하게 둠
        study.optimize(objective, n_trials=30)
    except KeyboardInterrupt:
        print("\n사용자에 의해 튜닝이 중단됨.")

    print("\n" + "="*50)
    print(f"🏆 Scene 0 전용 황금 베스트 하이퍼파라미터 🏆")
    print(f"최소 Score: {study.best_value:.4f}")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*50)