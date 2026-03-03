import cv2
import numpy as np

# ==========================================
# 이미지 파일 이름 (본인 PC에 있는 이미지)
IMG_PATH = './asset/scene/SCENARIO_1/rgb/0154.png' 
# ==========================================

points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"포인트 클릭됨: ({x}, {y})")
            cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Lane Picker", img_display)
            
            if len(points) == 4:
                print("\n[완료] 4개 포인트 선택됨!")
                print("키보드의 'q'를 누르면 저장하고 종료합니다. 'r'을 누르면 초기화합니다.")

img = cv2.imread(IMG_PATH)
if img is None:
    print(f"에러: {IMG_PATH} 파일을 찾을 수 없습니다.")
    exit()

img_display = img.copy()
cv2.namedWindow("Lane Picker")
cv2.setMouseCallback("Lane Picker", mouse_callback)

print("============ 사용법 ============")
print("1. 이미지가 뜨면 차선 위의 점 4개를 순서대로 클릭하세요.")
print("   순서: [왼쪽-위] -> [오른쪽-위] -> [왼쪽-아래] -> [오른쪽-아래]")
print("2. 4개를 다 찍으면 'q'를 눌러 저장하세요.")
print("3. 잘못 찍었으면 'r'을 눌러 다시 찍으세요.")
print("================================")

while True:
    cv2.imshow("Lane Picker", img_display)
    key = cv2.waitKey(1) & 0xFF
    
    # 'q' 키를 누르면 저장 후 종료
    if key == ord('q') and len(points) == 4:
        break
    # 'r' 키를 누르면 리셋
    elif key == ord('r'):
        points = []
        img_display = img.copy()
        print("초기화되었습니다. 다시 찍어주세요.")
    # 'ESC' 누르면 그냥 종료
    elif key == 27:
        exit()

cv2.destroyAllWindows()

# 결과 정리 및 CSV 포맷 출력
# 클릭 순서가 섞였을 수 있으니 Y좌표로 상/하 정렬 후, X좌표로 좌/우 정렬
# (상위 그룹 2개, 하위 그룹 2개 분리)
points.sort(key=lambda p: p[1]) # Y 기준 정렬
top_group = sorted(points[:2], key=lambda p: p[0]) # 상단 좌/우
bot_group = sorted(points[2:], key=lambda p: p[0]) # 하단 좌/우

p1 = top_group[0] # Top-Left
p2 = top_group[1] # Top-Right
p3 = bot_group[0] # Bot-Left
p4 = bot_group[1] # Bot-Right

print("\n========== 최종 좌표 ==========")
print(f"P1 (좌상): {p1}")
print(f"P2 (우상): {p2}")
print(f"P3 (좌하): {p3}")
print(f"P4 (우하): {p4}")

# CSV 파일 생성
csv_content = "filename,p1_x,p1_y,p2_x,p2_y,p3_x,p3_y,p4_x,p4_y\n"
csv_content += f"{IMG_PATH},{p1[0]},{p1[1]},{p2[0]},{p2[1]},{p3[0]},{p3[1]},{p4[0]},{p4[1]}\n"

with open("lane_points.csv", "w") as f:
    f.write(csv_content)

print("\n[성공] 'lane_points.csv' 파일이 저장되었습니다.")
