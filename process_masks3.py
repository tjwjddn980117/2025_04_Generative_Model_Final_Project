import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def process_masks_per_image(base_seg_dir,
                            output_dir,
                            exts=(".png", ".jpg", ".jpeg")):
    """
    base_seg_dir 아래 여러 서브폴더에 공통으로 들어있는 마스크 파일을 대상으로,
    각 픽셀의 모델별 값(0~255)을 0~1로 정규화한 뒤, 모델 수로 평균을 내서
    그 평균값(0~1)을 0~255 그레이스케일로 바꿔 저장합니다.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_dirs = sorted(
        d for d in os.listdir(base_seg_dir)
        if os.path.isdir(os.path.join(base_seg_dir, d))
    )
    # 공통 파일명
    file_sets = [
        {f for f in os.listdir(os.path.join(base_seg_dir, m))
         if f.lower().endswith(exts)}
        for m in model_dirs
    ]
    common_files = sorted(set.intersection(*file_sets))
    n_models = len(model_dirs)
    print(f"Found {n_models} models, {len(common_files)} common files.")

    for fname in tqdm(common_files, desc="Processing masks"):
        # 기준 크기
        ref = Image.open(os.path.join(base_seg_dir, model_dirs[0], fname)).convert("RGB")
        w, h = ref.size

        # 모델별 정규화된 그레이 맵 수집
        stack = np.zeros((n_models, h, w), dtype=np.float32)
        for i, m in enumerate(model_dirs):
            img = Image.open(os.path.join(base_seg_dir, m, fname)).convert("RGB")
            if img.size != (w, h):
                img = img.resize((w, h), resample=Image.NEAREST)
            arr = np.array(img, dtype=np.uint8)
            gray = arr.mean(axis=-1).astype(np.float32)  # 0~255
            stack[i] = gray / 255.0                      # 0~1

        # 픽셀별 평균 밝기 0~1
        mean_map = stack.mean(axis=0)

        # 0~1 → 0~255 uint8
        out_gray = (mean_map * 255).round().astype(np.uint8)

        # 1채널 그레이 이미지를 3채널로 복제
        out_arr = np.stack([out_gray]*3, axis=-1)  # shape (h,w,3)

        # 저장
        save_name = os.path.splitext(fname)[0] + ".png"
        save_path = os.path.join(output_dir, save_name)
        Image.fromarray(out_arr).save(save_path)


if __name__ == "__main__":
    seg_root = "seg_ex"   # 당신의 seg 폴더 경로
    output_dir = "C:\\Users\\Seo\\Desktop\\Gits\\ProjectPractice\\DiffV2IR-main\\processed"
    process_masks_per_image(seg_root, output_dir)