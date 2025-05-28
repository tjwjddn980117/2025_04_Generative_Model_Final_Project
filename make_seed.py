import json
import shutil
from pathlib import Path

def sync_and_seed(
    ir_folder:       str,
    rgb_folder:      str,
    seg_folder:      str,
    new_ir_folder:   str,
    new_rgb_folder:  str,
    new_seg_folder:  str,
    output_seed:     str = "seeds.json",
    rgb_exts         = (".jpg", ".jpeg", ".png"),
    ir_exts          = (".jpg", ".jpeg", ".png"),
    seg_exts         = (".png",),
):
    ir_src  = Path(ir_folder)
    rgb_src = Path(rgb_folder)
    seg_src = Path(seg_folder)

    ir_dst  = Path(new_ir_folder)
    rgb_dst = Path(new_rgb_folder)
    seg_dst = Path(new_seg_folder)

    # 대상 폴더 생성
    for d in (ir_dst, rgb_dst, seg_dst):
        d.mkdir(parents=True, exist_ok=True)

    # 1) 각 폴더의 stem 집합
    ir_stems  = {p.stem: p.suffix for p in ir_src.iterdir()  if p.is_file() and p.suffix.lower() in ir_exts}
    rgb_stems = {p.stem: p.suffix for p in rgb_src.iterdir() if p.is_file() and p.suffix.lower() in rgb_exts}
    seg_stems = {p.stem: p.suffix for p in seg_src.iterdir() if p.is_file() and p.suffix.lower() in seg_exts}

    # 2) 교집합 stem만 골라서 복사
    common = sorted(set(ir_stems) & set(rgb_stems) & set(seg_stems))
    seeds = []

    for stem in common:
        # 원본 파일 경로
        ir_file  = ir_src  / f"{stem}{ir_stems[stem]}"
        rgb_file = rgb_src / f"{stem}{rgb_stems[stem]}"
        seg_file = seg_src / f"{stem}{seg_stems[stem]}"

        # 대상 파일 경로 (확장자 그대로 유지)
        ir_dst_file  = ir_dst  / ir_file.name
        rgb_dst_file = rgb_dst / rgb_file.name
        seg_dst_file = seg_dst / seg_file.name

        # 복사
        shutil.copy2(ir_file,  ir_dst_file)
        shutil.copy2(rgb_file, rgb_dst_file)
        shutil.copy2(seg_file, seg_dst_file)

        # seeds.json 용 (RGB 확장자를 기준으로)
        seeds.append([rgb_file.name])

    # 3) JSON 저장
    with open(output_seed, "w", encoding="utf-8") as fp:
        json.dump(seeds, fp, ensure_ascii=False, indent=2)

    print(f"Copied {len(common)} files to new folders and saved seeds to {output_seed}.")


if __name__ == "__main__":
    # 원본 폴더
    ir_folder     = "C:\\Users\\Seo\\Desktop\\Gits\\ProjectPractice\\DiffV2IR-main\\Our_data\\ir"
    rgb_folder    = "C:\\Users\\Seo\\Desktop\\Gits\\ProjectPractice\\DiffV2IR-main\\Our_data\\rgb"
    seg_folder    = "C:\\Users\\Seo\\Desktop\\Gits\\ProjectPractice\\DiffV2IR-main\\Our_data\\seg"

    # 새로 복사할 폴더
    new_ir_folder  = "C:\\Users\\Seo\\Desktop\\Gits\\ProjectPractice\\DiffV2IR-main\\Our_data_new\\ir"
    new_rgb_folder = "C:\\Users\\Seo\\Desktop\\Gits\\ProjectPractice\\DiffV2IR-main\\Our_data_new\\rgb"
    new_seg_folder = "C:\\Users\\Seo\\Desktop\\Gits\\ProjectPractice\\DiffV2IR-main\\Our_data_new\\seg"

    sync_and_seed(
        ir_folder,
        rgb_folder,
        seg_folder,
        new_ir_folder,
        new_rgb_folder,
        new_seg_folder,
        output_seed="seeds.json"
    )