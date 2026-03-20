import optuna
import subprocess

SCENE_IDX = 0
EPOCH = 500

def objective(trial):
    lr = trial.suggest_float("learning_rate", 0.001, 0.04, log=True)
    lb = trial.suggest_float("lambda", 1e-8, 1e-5, log=True)

    # 저장될 이미지 이름을 위해 trial 이름표 생성
    trial_name = f"trial_{trial.number}_lr{lr:.4f}_lb{lb:.6f}"
    
    cmd = [
        "python", "run.py",
        "--depth_model", "monodepth2",
        "--idx", str(SCENE_IDX),
        "--update", "bim",
        "--learning_rate", str(lr),
        "--lambda", str(lb),
        "--epoch", str(EPOCH),
        "--train_log_flag",
        "--inner_eval_flag",
        "--patch_file", "6.jpg",
        "--scene_dir", "./asset/scene",
        "--patch_dir", "./asset/patch",
        "--obj_dir", "./asset/obstacle",
        "--csv_dir", "./asset/scene/scene_lane_points.csv",
        "--random_object_flag",
        "--random_test_flag",
        "--train_log_flag",
        "--train_offset_object_flag",
        "--test_offset_object_flag",
        "--train_offset_patch_flag",
        "--test_offset_patch_flag",
        "--train_color_patch_flag",
        "--test_color_patch_flag",
        "--train_color_object_flag",
        "--test_color_object_flag",
        "--train_quan_patch_flag",
        "--test_quan_patch_flag",
        "--log_dir_comment", trial_name
    ]
    
    print(f"\n[Trial {trial.number}] Testing: LR={lr:.4f}, Lambda={lb:.6f}")
    
    # 캡처를 켜서 run.py의 print 출력을 문자열로 받아옴
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
            print(f"🚨 [Error] 실행 중 튕김!")
            error_msg = result.stderr if result.stderr else result.stdout
            print(error_msg[-1000:])
            return 0.0, 0.0 
    
    mrsr = 0.0
    ssim = 0.0
    found_mrsr = False
    found_ssim = False
    
    # stdout 출력 중에서 우리가 심어둔 태그 찾기
    for line in result.stdout.split('\n'):
        if "OPTUNA_FINAL_MRSR:" in line:
            mrsr = float(line.split(":")[1].strip())
            found_mrsr = True
        if "OPTUNA_FINAL_SSIM:" in line:
            ssim = float(line.split(":")[1].strip())
            found_ssim = True
            
    # 태그를 성공적으로 찾았다면
    if found_mrsr and found_ssim:
        print(f"Result -> Final MRSR: {mrsr:.4f} | Final SSIM: {ssim:.4f}")
        return mrsr, ssim  # ✅ 두 개의 목적 함수 값 반환!
            
    # 값을 못 찾았을 경우
    print("⚠️ MRSR/SSIM 태그를 출력에서 찾지 못했습니다. run.py 코드를 확인하세요.")
    return 0.0, 0.0

if __name__ == "__main__":
    # ✅ 두 지표(MRSR, SSIM) 모두 최대화(Maximize)하도록 설정
    study = optuna.create_study(directions=["maximize", "maximize"])
    
    # 실험 횟수 
    study.optimize(objective, n_trials=50) 
    
    print("\n" + "="*50)
    print("최적화 완료! (다중 목적 최적화 분석 결과)")
    
    # ✅ 네가 원하는 조건 필터링: MRSR >= 1.0 & SSIM >= 0.3
    sweet_spots = []
    
    # study.best_trials는 다중 목적 최적화에서 '파레토 프론트(Pareto Front)'에 해당하는 최고의 균형점들을 담고 있음
    for trial in study.best_trials:
        best_mrsr, best_ssim = trial.values
        if best_mrsr >= 1.0 and best_ssim >= 0.3:
            sweet_spots.append(trial)
            
    if sweet_spots:
        print(f"\n🎯 [목표 달성] MRSR >= 1.0 이고 SSIM >= 0.3 인 완벽한 균형점들:")
        for t in sweet_spots:
            print(f" - Trial {t.number} | MRSR: {t.values[0]:.4f} | SSIM: {t.values[1]:.4f} | Params: {t.params}")
    else:
        print("\n⚠️ 목표 조건(MRSR>=1.0 & SSIM>=0.3)을 완전히 만족하는 조합은 찾지 못했습니다.")
        print("하지만 획득한 가장 좋은 타협점(Pareto Front)들은 다음과 같습니다:")
        for t in study.best_trials:
            print(f" - Trial {t.number} | MRSR: {t.values[0]:.4f} | SSIM: {t.values[1]:.4f} | Params: {t.params}")