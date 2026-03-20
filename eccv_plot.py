import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison():
    # 자동으로 저장된 CSV 파일 경로
    baseline_csv = "runs/debug_eot_scene/Baseline_Single_robustness.csv"
    ours_csv = "runs/tune_trial_12/debug_eot_scene/Ours_EoT_robustness.csv"
    
    if not os.path.exists(baseline_csv) or not os.path.exists(ours_csv):
        print("❌ CSV 파일이 없습니다. Baseline과 Ours 학습을 먼저 한 번씩 돌려주세요!")
        return

    # 데이터 불러오기
    df_base = pd.read_csv(baseline_csv)
    df_ours = pd.read_csv(ours_csv)

    # 🌟 [추가] 데이터가 거리 순서대로 있지 않을 경우를 대비해 오름차순 정렬
    df_base = df_base.sort_values(by='Distance')
    df_ours = df_ours.sort_values(by='Distance')

    # 논문용 고품질 그래프 그리기
    plt.figure(figsize=(10, 6)) # 크기를 조금 더 키움
    
    # Baseline (빨간색)
    plt.plot(df_base['Distance'], df_base['MRSR'], 
             label='w/o EoT (AdvRM Baseline)', color='#e74c3c', linewidth=2.5, marker='x', markersize=8)
    
    # Ours (파란색)
    plt.plot(df_ours['Distance'], df_ours['MRSR'], 
             label='w/ Perspective EoT (Ours)', color='#3498db', linewidth=2.5, marker='o', markersize=8)
    
    # X축 범위를 10m~35m로 명시적 설정 및 간격 조정
    plt.xticks(range(10, 36, 2)) # 2m 간격으로 눈금 표시
    plt.xlim(10, 35)

    # 그래프 꾸미기 (폰트 및 그리드)
    plt.xlabel('Distance to Obstacle (m)', fontsize=13, fontweight='bold')
    plt.ylabel('Mean Relative Shift Ratio (MRSR)', fontsize=13, fontweight='bold')
    plt.title('Robustness Comparison Over Continuous Distances', fontsize=15, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 범례 위치를 데이터에 방해받지 않는 곳으로 설정
    plt.legend(fontsize=12, loc='best')
    
    # 저장 및 출력
    save_path = "runs/debug_eot_scene/Final_Paper_Graph_Ordered.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=400) # 논문용 고해상도 저장
    plt.show() # 화면에도 즉시 표시
    print(f"✅ 정렬된 비교 그래프가 생성되었습니다: {save_path}")
    
if __name__ == "__main__":
    plot_comparison()
