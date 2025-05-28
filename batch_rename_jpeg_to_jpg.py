import os
from pathlib import Path

def batch_rename_jpeg_to_jpg(folder: str):
    folder = Path(folder)
    # 1) *.jpeg 파일 모으기
    jpeg_files = list(folder.glob("*.jpeg"))
    print(f"Found {len(jpeg_files)} .jpeg files in {folder}")

    # 2) 하나씩 .jpg 로 이름 변경
    for old_path in jpeg_files:
        new_path = old_path.with_suffix(".jpg")
        if new_path.exists():
            # print(f"  [SKIP] {new_path.name} already exists.")
            continue
        # print(f"  Renaming {old_path.name} -> {new_path.name}")
        old_path.rename(new_path)

    print("Done.")

if __name__ == "__main__":
    ir_folder = "C:\\Users\\Seo\\Desktop\\Gits\\ProjectPractice\\DiffV2IR-main\\Our_data\\ir"
    batch_rename_jpeg_to_jpg(ir_folder)