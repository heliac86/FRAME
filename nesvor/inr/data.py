from typing import Dict, List, Optional
import torch
from ..utils import gaussian_blur
from ..transform import RigidTransform, transform_points
from ..image import Volume, Slice

# ===== [E3] 패치 유효율 임계값 =====
# 패치 내 유효 픽셀(뇌 실질) 비율이 이 값 미만이면 해당 패치를 skip.
# 배경이 대부분인 패치가 FF Loss에 투입되어 주파수 스펙트럼을 오염시키는 것을 방지.
PATCH_VALID_RATIO_THRESHOLD = 0.5
# ===== [E3 끝] =====


class PointDataset(object):
    def __init__(self, slices: List[Slice]) -> None:
        self.mask_threshold = 1  # args.mask_threshold

        xyz_all = []
        v_all = []
        slice_idx_all = []
        transformation_all = []
        resolution_all = []

        # ===== [FF Loss 추가] 패치 샘플링을 위한 슬라이스별 2D 데이터 보존 =====
        self.slice_images: List[torch.Tensor] = []
        self.slice_masks: List[torch.Tensor] = []
        self.slice_shape_xyz: List[torch.Tensor] = []
        self.slice_resolution_xyz: List[torch.Tensor] = []
        # ===== [FF Loss 추가 끝] =====

        # ===== [HM2] 슬라이스별 픽셀 인덱스 범위 저장 (get_batch hard mining용) =====
        self._slice_pixel_ranges: List[tuple] = []  # (start, end) inclusive range per slice
        # ===== [HM2 끝] =====

        pixel_offset = 0
        for i, slice in enumerate(slices):
            xyz = slice.xyz_masked_untransformed
            v = slice.v_masked
            n_pixels = v.shape[0]
            slice_idx = torch.full(v.shape, i, device=v.device)
            xyz_all.append(xyz)
            v_all.append(v)
            slice_idx_all.append(slice_idx)
            transformation_all.append(slice.transformation)
            resolution_all.append(slice.resolution_xyz)

            # ===== [FF Loss 추가] 슬라이스 2D 데이터 저장 =====
            self.slice_images.append(slice.image[0].detach().clone())
            self.slice_masks.append(slice.mask[0].detach().clone())
            self.slice_shape_xyz.append(slice.shape_xyz.clone())
            self.slice_resolution_xyz.append(slice.resolution_xyz.clone())
            # ===== [FF Loss 추가 끝] =====

            # ===== [HM2] 픽셀 인덱스 범위 기록 =====
            self._slice_pixel_ranges.append((pixel_offset, pixel_offset + n_pixels))
            pixel_offset += n_pixels
            # ===== [HM2 끝] =====

        self.xyz = torch.cat(xyz_all)
        self.v = torch.cat(v_all)
        self.slice_idx = torch.cat(slice_idx_all)
        self.transformation = RigidTransform.cat(transformation_all)
        self.resolution = torch.stack(resolution_all, 0)
        self.count = self.v.shape[0]
        self.epoch = 0

        # ===== [HM2] 슬라이스별 픽셀 수 텐서 (가중 샘플링 계산용) =====
        self._slice_pixel_counts = torch.tensor(
            [e - s for s, e in self._slice_pixel_ranges], dtype=torch.float32
        )
        # ===== [HM2 끝] =====

    @property
    def n_slices(self) -> int:
        return len(self.slice_images)

    @property
    def bounding_box(self) -> torch.Tensor:
        max_r = self.resolution.max()
        xyz_transformed = self.xyz_transformed
        xyz_min = xyz_transformed.amin(0) - 2 * max_r
        xyz_max = xyz_transformed.amax(0) + 2 * max_r
        bounding_box = torch.stack([xyz_min, xyz_max], 0)
        return bounding_box

    @property
    def mean(self) -> float:
        q1, q2 = torch.quantile(
            self.v if self.v.numel() < 256 * 256 * 256 else self.v[: 256 * 256 * 256],
            torch.tensor([0.1, 0.9], dtype=self.v.dtype, device=self.v.device),
        )
        return self.v[torch.logical_and(self.v > q1, self.v < q2)].mean().item()

    # ===== [k_norm] 훈련 타겟 일괄 정규화 =====
    def normalize(self, v_mean: float) -> None:
        if v_mean <= 0:
            raise ValueError(f"v_mean must be positive, got {v_mean}")
        self.v = self.v / v_mean
        self.slice_images = [
            img / v_mean for img in self.slice_images
        ]
    # ===== [k_norm 끝] =====

    def get_batch(
        self,
        batch_size: int,
        device,
        sampling_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """메인 MSE 학습용 픽셀 배치 반환.

        Args:
            batch_size: 배치 크기.
            device: 반환 텐서를 배치할 디바이스.
            sampling_probs: 슬라이스별 샘플링 확률 (n_slices,) CPU 텐서.
                None이면 기존 randperm 균등 셔플.
                주어지면 에포크 경계마다 슬라이스 픽셀 수에 비례하여 가중
                샘플링된 순서로 전체 풀을 재정렬 (Hard Slice Mining).
        """
        if self.count + batch_size > self.xyz.shape[0]:
            self.count = 0
            self.epoch += 1

            if sampling_probs is None:
                # 기존 동작: 균등 무작위 셔플
                idx = torch.randperm(self.xyz.shape[0], device=device)
            else:
                # ===== [HM2] 가중 샘플링: 슬라이스 잔차에 비례해 픽셀을 재정렬 =====
                # 1) 슬라이스별 픽셀 가중치 = sampling_probs * n_pixels_in_slice
                #    (슬라이스를 더 자주 보이되, 각 슬라이스 내부는 균등하게 커버)
                pixel_weights = sampling_probs * self._slice_pixel_counts
                pixel_weights = pixel_weights / pixel_weights.sum()

                # 2) 슬라이스별 인덱스 블록을 가중치에 따라 multinomial 재정렬
                #    전체 픽셀 수만큼 replacement=True로 샘플링 후 사용
                #    (replacement=False는 수백만 원소에 느리므로 대신
                #     슬라이스 단위 셔플 + 내부 randperm 조합 사용)
                n_total = self.xyz.shape[0]
                slice_order = torch.multinomial(
                    pixel_weights, self.n_slices, replacement=False
                )  # (n_slices,) 슬라이스 방문 순서

                parts = []
                for s in slice_order.tolist():
                    start, end = self._slice_pixel_ranges[s]
                    perm = torch.randperm(end - start, device=device) + start
                    parts.append(perm)
                idx = torch.cat(parts)  # 길이 = n_total (모든 픽셀 한 번씩)
                # ===== [HM2 끝] =====

            self.xyz = self.xyz[idx]
            self.v = self.v[idx]
            self.slice_idx = self.slice_idx[idx]

        batch = {
            "xyz": self.xyz[self.count : self.count + batch_size],
            "v": self.v[self.count : self.count + batch_size],
            "slice_idx": self.slice_idx[self.count : self.count + batch_size],
        }
        self.count += batch_size
        return batch

    # ===== [FF Loss 추가] 패치 단위 샘플링 메서드 =====
    def get_patch_batch(
        self,
        n_patches: int,
        patch_size: int,
        device,
        sampling_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """FF Loss 계산을 위한 패치 배치 샘플링.

        Args:
            n_patches: 샘플링할 패치 수.
            patch_size: 패치의 한 변 크기 P (P×P 패치).
            device: 출력 텐서를 배치할 디바이스.
            sampling_probs: 슬라이스별 샘플링 확률 (n_slices,) CPU 텐서.
                None이면 균등 무작위 샘플링.
                Hard Slice Mining 시 slice_residuals EMA를 정규화하여 전달.

        Returns:
            dict:
                "xyz_patch"        : (n, P*P, 3)  untransformed 3D 좌표
                "v_patch"          : (n, P, P)    GT 픽셀 강도값
                "slice_idx_patch"  : (n,)         패치가 속한 슬라이스 인덱스
                "valid_mask_patch" : (n, P, P)    유효 픽셀 bool 마스크
            패치보다 작은 슬라이스만 존재하거나 유효 패치가 없으면 빈 dict 반환.
        """
        P = patch_size
        n_slices = len(self.slice_images)

        xyz_list = []
        v_list = []
        sidx_list = []
        valid_list = []

        # ===== [E3] 패치 유효율 필터링: 최대 시도 횟수 설정 =====
        # 유효율 미달 패치를 skip하므로 n_patches개를 채우지 못할 수 있음.
        # 무한 루프 방지를 위해 최대 시도 횟수를 n_patches * 5로 제한.
        max_attempts = n_patches * 5
        attempts = 0
        collected = 0
        # ===== [E3 끝] =====

        while collected < n_patches and attempts < max_attempts:
            attempts += 1

            # ===== [Hard Slice Mining] 슬라이스 선택 =====
            # sampling_probs가 주어지면 가중 샘플링, 아니면 균등 무작위
            if sampling_probs is not None:
                s_idx = int(torch.multinomial(sampling_probs, 1).item())
            else:
                s_idx = int(torch.randint(n_slices, (1,)).item())
            # ===== [Hard Slice Mining 끝] =====

            img = self.slice_images[s_idx]
            msk = self.slice_masks[s_idx]
            shape_xyz = self.slice_shape_xyz[s_idx]
            res_xyz = self.slice_resolution_xyz[s_idx]

            H, W = img.shape
            if H < P or W < P:
                continue

            r0 = int(torch.randint(0, H - P + 1, (1,)).item())
            c0 = int(torch.randint(0, W - P + 1, (1,)).item())

            valid_patch = msk[r0 : r0 + P, c0 : c0 + P]

            # ===== [E3] 유효율 필터링 =====
            # 패치 내 유효 픽셀 비율이 임계값 미만이면 skip.
            # mean-fill과 함께 작동: 유효 픽셀이 충분해야 fill 평균값이 신뢰할 수 있음.
            if valid_patch.float().mean().item() < PATCH_VALID_RATIO_THRESHOLD:
                continue
            # ===== [E3 끝] =====

            v_patch = img[r0 : r0 + P, c0 : c0 + P].to(device)
            valid_patch = valid_patch.to(device)

            rows = torch.arange(r0, r0 + P, dtype=torch.float32, device=device)
            cols = torch.arange(c0, c0 + P, dtype=torch.float32, device=device)
            row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
            z_grid = torch.zeros_like(row_grid)

            kji = torch.stack([col_grid, row_grid, z_grid], dim=-1)
            shape_xyz_dev = shape_xyz.to(device).float()
            res_dev = res_xyz.to(device).float()
            xyz_patch = (kji - (shape_xyz_dev - 1) / 2) * res_dev

            xyz_list.append(xyz_patch.view(P * P, 3))
            v_list.append(v_patch)
            sidx_list.append(
                torch.full((1,), s_idx, dtype=torch.long, device=device)
            )
            valid_list.append(valid_patch)
            collected += 1

        if len(xyz_list) == 0:
            return {}

        return {
            "xyz_patch":         torch.stack(xyz_list, dim=0),
            "v_patch":           torch.stack(v_list, dim=0),
            "slice_idx_patch":   torch.cat(sidx_list, dim=0),
            "valid_mask_patch":  torch.stack(valid_list, dim=0),
        }
    # ===== [FF Loss 추가 끝] =====

    @property
    def xyz_transformed(self) -> torch.Tensor:
        return transform_points(self.transformation[self.slice_idx], self.xyz)

    @property
    def mask(self) -> Volume:
        with torch.no_grad():
            resolution_min = self.resolution.min()
            resolution_max = self.resolution.max()
            xyz = self.xyz_transformed
            xyz_min = xyz.amin(0) - resolution_max * 10
            xyz_max = xyz.amax(0) + resolution_max * 10
            shape_xyz = ((xyz_max - xyz_min) / resolution_min).ceil().long()
            shape = (int(shape_xyz[2]), int(shape_xyz[1]), int(shape_xyz[0]))
            kji = ((xyz - xyz_min) / resolution_min).round().long()

            mask = torch.bincount(
                kji[..., 0]
                + shape[2] * kji[..., 1]
                + shape[2] * shape[1] * kji[..., 2],
                minlength=shape[0] * shape[1] * shape[2],
            )
            mask = mask.view((1, 1) + shape).float()
            mask_threshold = (
                self.mask_threshold
                * resolution_min**3
                / self.resolution.log().mean().exp() ** 3
            )
            mask_threshold *= mask.sum() / (mask > 0).sum()
            assert len(mask.shape) == 5
            mask = (
                gaussian_blur(mask, (resolution_max / resolution_min).item(), 3)
                > mask_threshold
            )[0, 0]

            xyz_c = xyz_min + (shape_xyz - 1) / 2 * resolution_min
            return Volume(
                mask.float(),
                mask,
                RigidTransform(torch.cat([0 * xyz_c, xyz_c])[None], True),
                resolution_min,
                resolution_min,
                resolution_min,
            )
