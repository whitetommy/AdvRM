import optuna
import os
import pandas as pd
from option import get_args
from advrm import ADVRM
from torch.utils.tensorboard import SummaryWriter

def objective(trial):
    args = get_args()
    
    args['learning_rate'] = trial.suggest_float('learning_rate', 1e-3, 5e-3, log=True)
    args['lambda'] = trial.suggest_float('lambda', 1e-6, 1e-5, log=True)
    args['tv_weight'] = trial.suggest_float('tv_weight', 1e-6, 2e-5, log=True)
    
    args['inner_eval_flag'] = True 

    print(f"\n🚀 [Trial {trial.number}] 시작! 하이퍼파라미터: {trial.params}")

    csv_file = args['csv_dir']
    scene_set = pd.read_csv(csv_file)
    idx = args['idx']
    
    scene_file = scene_set.iloc[idx, 0]
    secen_key_points = scene_set.iloc[idx]
    
    log_dir = os.path.join(args['log_dir'], f"trial_{trial.number}")
    logger = SummaryWriter(log_dir)
    args['log_dir'] = log_dir

    p_size = args['patch_size'] if args['patch_size'] != "" else None
    
    advrm = ADVRM(args, log=logger, patch_size=p_size, rewrite=True, random_object_flag=args['random_object_flag'])
    
    best_loss, best_ssim, final_mrsr = advrm.run(args['scene_dir'], scene_file, idx, secen_key_points)
    
    if best_ssim < 0.3:
        print(f"탈락: 은밀성 완전 파괴 (SSIM: {best_ssim:.4f})")
        return -9999.0  # (최대화 목표이므로 최악의 점수 부여)
    
    if final_mrsr < 0.0:
        print(f"탈락: 공격력 부족 (Test MRSR: {final_mrsr:.4f})")
        return -9999.0

    print(f"합격 후보! SSIM: {best_ssim:.4f} / Test MRSR: {final_mrsr:.4f}")
    
    return final_mrsr


def main():
    study = optuna.create_study(
        study_name="AdvRM_MRSR_Maximization", 
        direction="maximize"
    )
    # 최적화 실행 (50번의 탐색 진행)
    study.optimize(objective, n_trials=20)
    
    # 결과 터미널 출력
    print("\n==================================================")
    print("🎉 하이퍼파라미터 튜닝 완료!")
    print(f"🥇 Best Trial Number: {study.best_trial.number}")
    print(f"🏆 획득한 최고 Test MRSR: {study.best_value:.4f}")
    print("🥇 Best Parameters:")
    for key, value in study.best_trial.params.items():
        print(f"    --{key}: {value}")
    print("==================================================")

    # 결과를 텍스트 파일로 안전하게 저장
    with open("best_hyperparameters.txt", "w") as f:
        f.write("🎉 AdvRM Hyperparameter Tuning Results (MRSR Maximization) 🎉\n")
        f.write("==================================================\n")
        f.write(f"Best Trial Number: {study.best_trial.number}\n")
        f.write(f"🏆 최고 달성 Test MRSR: {study.best_value:.4f}\n")
        f.write("Best Parameters:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"    --{key}: {value}\n")
        f.write("==================================================\n")
    print("💾 최적의 하이퍼파라미터가 'best_hyperparameters.txt'에 저장되었습니다.")

if __name__ == "__main__":
    main()