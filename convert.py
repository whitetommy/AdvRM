from PIL import Image

# 1. 현재 저장된 저해상도(200x150) 최적화 패치 불러오기
img_path = 'runs/final_patches/6_SCENARIO_6_rgb_0067_0_best.png'
img = Image.open(img_path)

# 2. 크기를 10배(2000x1500)로 키우기!
# 🌟 핵심: Image.NEAREST (또는 Image.Resampling.NEAREST)를 써서 픽셀을 뭉개지 않고 블록 그대로 복제
high_res_size = (600, 450) 
img_high_res = img.resize(high_res_size, Image.NEAREST)

# 3. 언리얼 엔진 임포트용으로 저장
save_path = 'runs/final_patches/patch_unreal_ready.png'
img_high_res.save(save_path)
print(f"언리얼 엔진용 고해상도 패치 저장 완료: {save_path}")