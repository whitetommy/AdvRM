import os
import glob
import pandas as pd
# from PIL import Image # 이미지 크기 읽을 필요 없음 (강제 고정)

# 1. 설정
scene_root = './asset/scene'
output_csv = './asset/scene/scene_lane_points.csv'

# 핵심 수정 사항: 코드 내부에서 사용하는 강제 해상도(600x450)에 맞춤
TARGET_W = 600
TARGET_H = 450

# 이미지 파일 찾기
search_pattern = os.path.join(scene_root, 'SCENARIO_*', 'rgb', '*.png')
image_paths = sorted(glob.glob(search_pattern))

print(f"Found {len(image_paths)} images. Generating coordinates for fixed size {TARGET_W}x{TARGET_H}...")

data = []

for path in image_paths:
    try:
        # 실제 이미지 크기와 상관없이, 코드가 인식하는 600x450 기준으로 좌표 생성
        
        # [사다리꼴 모양 잡기] - 600x450 캔버스 기준 비율
        # x1, y1: 왼쪽 위 (멀리)
        x1 = int(TARGET_W * 0.35)   # 210
        y1 = int(TARGET_H * 0.60)   # 270
        
        # x2, y2: 오른쪽 위 (멀리)
        x2 = int(TARGET_W * 0.65)   # 390
        y2 = int(TARGET_H * 0.60)   # 270
        
        # x3, y3: 오른쪽 아래 (가까이)
        x3 = int(TARGET_W * 0.85)   # 510 (이제 600 안쪽으로 들어옴!)
        y3 = int(TARGET_H * 0.90)   # 405
        
        # x4, y4: 왼쪽 아래 (가까이)
        x4 = int(TARGET_W * 0.15)   # 90
        y4 = int(TARGET_H * 0.90)   # 405

        # 경로 정리
        rel_path = os.path.relpath(path, scene_root).replace('\\', '/')
        
        row = {
            'Filename': rel_path,
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2,
            'x3': x3, 'y3': y3,
            'x4': x4, 'y4': y4
        }
        data.append(row)
        
    except Exception as e:
        print(f"Error processing {path}: {e}")

# 4. CSV 저장
if data:
    df = pd.DataFrame(data)
    cols = ['Filename', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
    df = df[cols]
    df.to_csv(output_csv, index=False)
    print(f"✅ Success! Generated CSV compatible with internal resolution.")
    print(f"Sample Coordinate (x3 should be < 600): {data[0]['x3']}")
else:
    print("❌ No images found!")
