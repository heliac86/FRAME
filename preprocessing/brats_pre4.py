import os
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

def degrade_mri_volume(data: np.ndarray, factor: int, sigma: float, gain_min: float, gain_max: float, is_seg: bool) -> np.ndarray:
    """
    물리적 수식 y = ((x * PSF) ↓_s) * G_alt 을 구현합니다. (노이즈 주입 제거)
    G_alt는 홀수/짝수 슬라이스에 따라 교차로 밝기를 올리고 내리는 Gain입니다.
    """
    if is_seg:
        # 세그멘테이션: 정수형 유지, 블러 생략, 단순 다운샘플링만 수행
        subsampled = data[:, :, ::factor]
        return subsampled.astype(np.uint8)

    # 1. PSF 시뮬레이션 (Z축 방향 1D 가우시안 블러)
    blurred_data = gaussian_filter1d(data, sigma=sigma, axis=2)
    # 2. 다운샘플링 (Z축 방향 건너뛰기)
    subsampled_data = blurred_data[:, :, ::factor]
    # 3. 교차 슬라이스 강도 변형 (Alternating Slice Intensity Variation)
    num_slices = subsampled_data.shape[2]
    slice_gains = np.ones(num_slices, dtype=np.float32)
    for i in range(num_slices):
        # gain_min(0.05) ~ gain_max(0.10) 사이의 난수 생성
        variation = np.random.uniform(gain_min, gain_max)
        if i % 2 == 0:
            # 파이썬 인덱스 0, 2, 4... (실제 1, 3, 5번째인 홀수 슬라이스) -> 밝게
            slice_gains[i] = 1.0 + variation
        else:
            # 파이썬 인덱스 1, 3, 5... (실제 2, 4, 6번째인 짝수 슬라이스) -> 어둡게
            slice_gains[i] = 1.0 - variation
    # 각 슬라이스에 gain 곱하기 (Broadcasting 활용)
    subsampled_data = subsampled_data * slice_gains
    return subsampled_data.astype(np.float32)

def process_single_file(in_path: Path, out_path: Path, config: dict):
    basename = in_path.name.lower()
    is_seg = '_seg' in basename or 'seg.' in basename
    img = nib.load(str(in_path))
    affine = img.affine.copy()
    header = img.header.copy()
    data = img.get_fdata(dtype=np.float32)
    degraded_data = degrade_mri_volume(
        data=data,
        factor=config['factor'],
        sigma=config['sigma'],
        gain_min=config['gain_min'],
        gain_max=config['gain_max'],
        is_seg=is_seg
    )
    affine[:3, 2] *= float(config['factor'])
    zooms = list(header.get_zooms())
    zooms[2] = zooms[2] * config['factor']
    header.set_zooms(tuple(zooms[:degraded_data.ndim]))
    new_img = nib.Nifti1Image(degraded_data, affine, header=header)
    new_img.set_sform(affine, code=1)
    new_img.set_qform(affine, code=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(new_img, str(out_path))

def main():
    parser = argparse.ArgumentParser(description="BraTS 물리적 성능 저하 파이프라인 (교차 슬라이스 강도 변동, 노이즈 제외)")
    parser.add_argument("--input_dir", required=True, help="원본 BraTS 최상위 폴더 경로")
    parser.add_argument("--output_dir", required=True, help="손상된 BraTS가 저장될 경로")
    parser.add_argument("--factor", type=int, default=4, help="Z축 다운샘플링 배율 (기본: 4)")
    parser.add_argument("--sigma", type=float, default=1.5, help="Z축 가우시안 블러 시그마 (기본: 1.5)")
    parser.add_argument("--gain_min", type=float, default=0.05, help="밝기 변동 최소치 (기본: 5%)")
    parser.add_argument("--gain_max", type=float, default=0.10, help="밝기 변동 최대치 (기본: 10%)")
    parser.add_argument("--seed", type=int, default=None, help="재현성을 위한 랜덤 시드")
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    config = {
        'factor': args.factor,
        'sigma': args.sigma,
        'gain_min': args.gain_min,
        'gain_max': args.gain_max
    }
    all_files = list(input_root.rglob("*.nii")) + list(input_root.rglob("*.nii.gz"))
    print(f"총 {len(all_files)}개의 파일을 찾았습니다.")
    print(f"적용 모델: y = ((x * PSF) ↓_s) * G_alt (노이즈 제외)")
    for file_path in tqdm(all_files, desc="Processing MRI files"):
        try:
            relative_path = file_path.relative_to(input_root)
            out_path = output_root / relative_path
            process_single_file(file_path, out_path, config)
        except Exception as e:
            print(f"\n❌ 에러 발생 ({file_path.name}): {str(e)}")

if __name__ == "__main__":
    main()