import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# base_run_dir = 'saved_eval_images/2026-04-07-22-06-27_' 
base_run_dir = 'saved_eval_images/2026-04-09-12-31-14_' 

# csv_path = 'runs/2026-04-07-22-06-27_/experiment_summary.csv'
csv_path = 'runs/2026-04-09-12-31-14_/experiment_summary.csv'

df = pd.read_csv(csv_path)
df_data = df[df['Scene_Idx'] != 'TOTAL_AVG'].copy()
df_data['Avg_E_Blend'] = df_data['Avg_E_Blend'].astype(float)

top_10 = df_data.sort_values(by='Avg_E_Blend', ascending=True).head(10)
bottom_10 = df_data.sort_values(by='Avg_E_Blend', ascending=False).head(10)
output_csv_dir = os.path.dirname(csv_path)

cols_to_save = ['Scene_Idx', 'Scene_Name', 'Avg_E_Blend', 'Avg_E_Cover', 'SSIM']
top_10[cols_to_save].to_csv(os.path.join(output_csv_dir, 'top_10_scenes.csv'), index=False, encoding='utf-8-sig')
bottom_10[cols_to_save].to_csv(os.path.join(output_csv_dir, 'bottom_10_scenes.csv'), index=False, encoding='utf-8-sig')
print(f"CSV 저장 완료: {output_csv_dir}")

def save_group_mosaic_v4(dataframe, group_name):
    num_scenes = len(dataframe)
    fig, axes = plt.subplots(num_scenes, 4, figsize=(22, 2.8 * num_scenes))
    
    img_names = ['ben_scene.png', 'ben_depth.png', 'adv_scene.png', 'adv_depth.png']
    img_labels = ['(A) Ben Scene', '(B) Ben Depth', '(C) Adv Scene', '(D) Adv Depth']

    TOP = 0.95
    BOTTOM = 0.05
    HSPACE = 0.25 

    for i, (idx, row) in enumerate(dataframe.iterrows()):
        scene_idx_num = int(row['Scene_Idx'])
        search_pattern = os.path.join(base_run_dir, f"scene_{scene_idx_num:03d}", 'epoch_1000', 'obj_*')
        found_folders = glob.glob(search_pattern)
        
        if found_folders:
            folder = found_folders[0]
            for j in range(4):
                ax = axes[i, j]
                img_path = os.path.join(folder, img_names[j])
                if os.path.exists(img_path):
                    img = mpimg.imread(img_path)
                    ax.imshow(img)
                
                if j == 0:
                    info_text = f"RANK {i+1:02d} | Scene_Idx: {scene_idx_num} | E_Blend: {row['Avg_E_Blend']:.4f} | E_Cover: {row['Avg_E_Cover']:.4f}"
                    ax.text(0, 1.12, info_text, transform=ax.transAxes, 
                            fontsize=18, fontweight='bold', color='#1A1A1A', va='bottom')
                
                ax.set_xlabel(img_labels[j], fontsize=13, fontweight='medium', labelpad=8)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor('#EEEEEE')
                    spine.set_linewidth(1)
        else:
            for j in range(4): axes[i, j].axis('off')

    plt.subplots_adjust(left=0.05, right=0.95, bottom=BOTTOM, top=TOP, hspace=HSPACE, wspace=0.01)

    fig.canvas.draw() 
    for i in range(num_scenes - 1):
        bbox_current = axes[i, 0].get_position()
        bbox_next = axes[i+1, 0].get_position()
        
        line_y = (bbox_current.y0 + bbox_next.y1) / 2
        
        fig.add_artist(plt.Line2D([0.05, 0.95], [line_y, line_y], 
                                  transform=fig.transFigure, color='#CCCCCC', 
                                  linestyle='--', linewidth=1.5, alpha=0.8))

    save_filename = f"{group_name}_figure.png"
    plt.savefig(os.path.join(output_csv_dir, save_filename), dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"=> 이미지 생성 완료: {save_filename}")

# 실행
save_group_mosaic_v4(top_10, "top10")
save_group_mosaic_v4(bottom_10, "bottom10")