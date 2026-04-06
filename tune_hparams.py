import optuna
import subprocess
import sys

def objective(trial):
    # 1. 탐색할 하이퍼파라미터 공간 정의 (과거 실험 팩트 기반 범위 수정)
    lr = trial.suggest_float("learning_rate", 1e-3, 2e-2, log=True) # 0.01 근처를 잘 탐색하도록 조정
    
    # 람다(Style 비중): 기존 로그를 바탕으로 1e-7 ~ 1e-5 사이를 집중 탐색
    lam = trial.suggest_float("lambda", 1e-8, 1e-5, log=True) 
    
    # TV_weight: 노이즈 억제 (람다와 비슷한 스케일로 탐색)
    tv_weight = trial.suggest_float("tv_weight", 1e-7, 1e-4, log=True)

    # 2. 실행할 터미널 명령어 구성
    epochs = "1000" 
    
    cmd = [
        "python", "run.py",
        "--depth_model", "monodepth2",
        "--idx", "0",
        "--update", "adam",
        "--patch_file", "6.jpg",
        "--scene_dir", "./asset/scene",
        "--patch_dir", "./asset/patch",
        "--obj_dir", "./asset/obstacle",
        "--csv_dir", "./asset/scene/scene_lane_points.csv",
        "--epoch", epochs,
        "--lambda", str(lam),
        "--learning_rate", str(lr),
        "--tv_weight", str(tv_weight),
        "--train_log_flag", "--random_object_flag", "--inner_eval_flag",
        "--train_offset_object_flag", "--test_offset_object_flag",
        "--train_offset_patch_flag", "--test_offset_patch_flag",
        "--train_color_patch_flag", "--test_color_patch_flag",
        "--train_color_object_flag", "--test_color_object_flag",
        "--train_quan_patch_flag", "--test_quan_patch_flag",
        "--random_test_flag"
    ]

    print(f"\n[{trial.number}번째 Trial 시작] lr={lr:.6f}, lambda={lam:.2e}, tv={tv_weight:.2e}")
    
    # 3. 서브프로세스로 run.py 실행
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    e_blend = None
    ssim = None
    
    # 4. 실시간 로그 파싱
    for line in process.stdout:
        # 터미널에 진행 상황을 다 보고 싶으면 아래 주석 해제
        # print(line, end="") 
        
        if "OPTUNA_FINAL_E_BLEND:" in line:
            e_blend = float(line.strip().split(":")[-1])
        elif "OPTUNA_FINAL_SSIM:" in line:
            ssim = float(line.strip().split(":")[-1])
            
    process.wait()

    # 값 추출 실패 시 Trial 중단
    if e_blend is None or ssim is None:
        raise optuna.exceptions.TrialPruned("실행 중 오류 발생으로 최종 Metric을 찾을 수 없음.")

    print(f"[{trial.number}번째 결과] E_blend: {e_blend:.4f}, SSIM: {ssim:.4f}")

    # 5. 제약 조건 (Penalty)
    # 목표: SSIM 0.4 이상. 따라서 SSIM이 0.38 미만으로 너무 떨어지면 가차 없이 페널티 부여.
    if ssim < 0.3:
        # 페널티를 줄 때, SSIM이 낮을수록 더 큰 페널티를 줘서 Optuna가 방향을 잡게 도움
        penalty = (0.3 - ssim) * 10.0 
        return e_blend + penalty 
        
    # 제약 조건을 통과한 경우, E_blend 최소화를 목표로 반환
    return e_blend

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", study_name="AdvRM_Tuning_SSIM0.4")
    
    try:
        # 실험 횟수는 30번 (컴퓨팅 자원에 따라 조절 가능)
        study.optimize(objective, n_trials=100)
    except KeyboardInterrupt:
        print("\n사용자에 의해 튜닝이 중단됨.")

    print("\n" + "="*40)
    print("🏆 베스트 하이퍼파라미터 🏆")
    print(f"최소 E_blend: {study.best_value:.4f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        if key == "learning_rate":
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value:.2e}")
    print("="*40)
