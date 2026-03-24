import optuna
import subprocess
import sys

def objective(trial):
    # 1. 탐색할 하이퍼파라미터 공간 정의
    # 로그 스케일(log=True)로 넓은 범위를 촘촘하게 탐색해.
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    lam = trial.suggest_float("lambda", 1e-7, 1e-4, log=True)
    tv_weight = trial.suggest_float("tv_weight", 1e-7, 1e-4, log=True)
    update_method = trial.suggest_categorical("update", ["adam", "bim"])

    # 2. 실행할 터미널 명령어 구성
    # 튜닝 속도를 위해 일단 epoch를 1000에서 500 정도로 줄여서 경향성만 보는 걸 추천해. (필요시 1000으로 원상복구)
    epochs = "500" 
    
    cmd = [
        "python", "run.py",
        "--depth_model", "monodepth2",
        "--idx", "0",
        "--update", update_method,
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

    print(f"\n[{trial.number}번째 Trial 시작] lr={lr:.6f}, lambda={lam:.2e}, tv={tv_weight:.2e}, update={update_method}")
    
    # 3. 서브프로세스로 run.py 실행
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    bg_gap = None
    coverage = None
    
    # 4. 실시간 로그 파싱
    for line in process.stdout:
        # 진행 상황이 궁금하면 아래 주석을 풀어서 터미널에 띄울 수 있어
        # print(line, end="") 
        
        if "OPTUNA_FINAL_BG_GAP:" in line:
            bg_gap = float(line.strip().split(":")[-1])
        elif "OPTUNA_FINAL_COVERAGE:" in line:
            coverage = float(line.strip().split(":")[-1])
            
    process.wait()

    # 값 추출 실패 시 Trial 중단
    if bg_gap is None or coverage is None:
        raise optuna.exceptions.TrialPruned("실행 중 오류 발생으로 최종 Metric을 찾을 수 없음.")

    print(f"[{trial.number}번째 결과] E_blend: {bg_gap:.4f}, Coverage: {coverage:.4f}")

    # 5. 제약 조건 (Penalty)
    # E_cover가 너무 떨어지면(예: 95% 미만) 그 하이퍼파라미터는 버리도록 페널티 부여
    if coverage < 0.95:
        return bg_gap + 10.0
        
    # E_blend(BG_GAP)를 최소화하는 것이 목표이므로 이 값을 반환
    return bg_gap

if __name__ == "__main__":
    # 방향을 minimize(최소화)로 설정
    study = optuna.create_study(direction="minimize", study_name="AdvRM_Tuning")
    
    # 총 30번의 실험 진행 (컴퓨팅 파워에 맞춰 늘려도 됨)
    try:
        study.optimize(objective, n_trials=30)
    except KeyboardInterrupt:
        print("\n사용자에 의해 튜닝이 중단됨.")

    # 최종 결과 출력
    print("\n" + "="*40)
    print("🏆 베스트 하이퍼파라미터 🏆")
    print(f"최소 E_blend (BG_GAP): {study.best_value:.4f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*40)