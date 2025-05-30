import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def process_masks_per_image(base_seg_dir,
                            output_dir,
                            exts=(".png", ".jpg", ".jpeg"),
                            threshold=0.41):
    """
    base_seg_dir 아래 여러 서브폴더에 공통으로 들어있는 마스크 파일을 대상으로,
    각 픽셀의 모델별 값(0~255)을 0~1로 정규화한 뒤, 
    모델 수로 평균을 냈을 때 threshold 이상인 위치만 흰색(True)으로 처리합니다.

    - threshold: 0~1 사이의 float. 예를 들어 0.25면 “4개 모델 중 평균이 0.25 이상” → 
      적어도 1개(=0.25*4) 정도 밝기가 있어야 흰색으로 간주.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) 모델별(폴더별) 리스트
    model_dirs = sorted(
        d for d in os.listdir(base_seg_dir)
        if os.path.isdir(os.path.join(base_seg_dir, d))
    )

    # 2) 공통 파일명 추출
    file_sets = [
        {f for f in os.listdir(os.path.join(base_seg_dir, m))
         if f.lower().endswith(exts)}
        for m in model_dirs
    ]
    common_files = sorted(set.intersection(*file_sets))

    n_models = len(model_dirs)
    print(f"Found {n_models} models, {len(common_files)} common files.")

    for fname in tqdm(common_files, desc="Processing masks"):
        # --- 기준 크기 가져오기 ---
        base_path = os.path.join(base_seg_dir, model_dirs[0], fname)
        ref_img = Image.open(base_path).convert("RGB")
        w, h = ref_img.size

        # --- 모델별 픽셀값 수집(0~1로 정규화) ---
        stack = np.zeros((n_models, h, w), dtype=np.float32)
        for i, m in enumerate(model_dirs):
            img = Image.open(os.path.join(base_seg_dir, m, fname)).convert("RGB")
            if img.size != (w, h):
                img = img.resize((w, h), resample=Image.NEAREST)
            arr = np.array(img, dtype=np.uint8)          # (h, w, 3), 0~255
            gray = arr.mean(axis=-1).astype(np.float32)  # 채널 평균 → 0~255
            stack[i] = gray / 255.0                      # 0~1

        # --- 모델별 평균 계산 ---
        mean_map = stack.mean(axis=0)  # shape (h, w), 0~1

        # --- 임계치 이상 픽셀만 흰색으로 ---
        mask = mean_map >= threshold

        # --- 결과 이미지 생성 & 저장 ---
        out_arr = np.zeros((h, w, 3), dtype=np.uint8)
        out_arr[mask] = [255, 255, 255]

        save_name = os.path.splitext(fname)[0] + ".png"
        save_path = os.path.join(output_dir, save_name)
        Image.fromarray(out_arr).save(save_path)

if __name__ == "__main__":
    seg_root = "seg"   # 당신의 seg 폴더 경로
    output_dir = "C:\\Users\\Seo\\Desktop\\Gits\\ProjectPractice\\DiffV2IR-main\\processed_yonsei"
    process_masks_per_image(seg_root, output_dir)