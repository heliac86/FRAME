"""
losses.py
---------
Focal Frequency Loss (FFL) 구현 모듈.

참고 논문:
  Focal Frequency Loss for Image Reconstruction and Synthesis
  Lama et al., ICCV 2021
  https://arxiv.org/abs/2012.12821

NeSVoR 학습 루프에서 패치 단위 FF Loss로 사용된다.
models.py 에서 FF_LOSS 키로 참조하고,
train.py 의 loss_weights 딕셔너리에 가중치를 등록한다.
"""

import torch
import torch.nn as nn


# models.py 의 D_LOSS, S_LOSS 등과 동일한 네이밍 규칙
FF_LOSS = "FFLoss"


class FocalFrequencyLoss(nn.Module):
    """패치 단위 Focal Frequency Loss.

    예측 패치와 정답 패치를 2D FFT로 주파수 공간으로 변환한 뒤,
    재현하기 어려운(hard) 주파수 성분에 더 높은 가중치를 부여하여
    손실을 계산한다.

    Args:
        alpha (float): 동적 가중치의 지수. 클수록 hard 주파수를 더 강조.
                       원 논문 권장값: 1.0
        patch_factor (int): 패치를 더 작게 분할할 배율. 1이면 분할 없음.
        ave_spectrum (bool): True이면 배치 평균 스펙트럼으로 손실 계산.
        log_matrix (bool): True이면 주파수 진폭에 로그 스케일 적용.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        patch_factor: int = 1,
        ave_spectrum: bool = False,
        log_matrix: bool = False,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix

    def tensor2freq(self, x: torch.Tensor) -> torch.Tensor:
        """2D FFT 변환 후 실수/허수 파트를 마지막 차원으로 stack.

        Args:
            x: (B, H, W) 형태의 패치 텐서.

        Returns:
            (B, H, W, 2) 형태. 마지막 차원 [0]=real, [1]=imag.
        """
        # ortho 정규화: 순전파/역전파 모두 에너지 보존
        freq = torch.fft.fft2(x, norm="ortho")
        # fftshift: DC(저주파) 성분을 중앙으로 이동
        freq = torch.fft.fftshift(freq, dim=(-2, -1))
        return torch.stack([freq.real, freq.imag], dim=-1)

    def _compute_loss(
        self,
        pred_freq: torch.Tensor,
        gt_freq: torch.Tensor,
    ) -> torch.Tensor:
        """동적 가중치를 이용한 주파수 공간 손실 계산.

        Args:
            pred_freq: (B, H, W, 2) 예측 주파수 맵.
            gt_freq:   (B, H, W, 2) 정답 주파수 맵.

        Returns:
            scalar 손실값.
        """
        # 주파수별 제곱 오차: (B, H, W)
        sq_diff = (pred_freq - gt_freq).pow(2).sum(dim=-1)

        # 동적 가중치: hard 주파수(오차가 큰 곳)에 높은 가중치 부여
        # detach() 로 가중치 자체는 역전파에서 제외
        weight = sq_diff.detach().pow(self.alpha / 2.0)
        # 배치 평균으로 정규화하여 스케일 안정화
        weight = weight / (weight.mean() + 1e-8)

        return (weight * sq_diff).mean()

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        """FF Loss 순전파.

        Args:
            pred: (B, H, W) 또는 (B, 1, H, W) 형태의 예측 패치.
            gt:   (B, H, W) 또는 (B, 1, H, W) 형태의 정답 패치.

        Returns:
            scalar FF Loss 값.
        """
        # 채널 차원 제거: (B, 1, H, W) → (B, H, W)
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if gt.dim() == 4:
            gt = gt.squeeze(1)

        B, H, W = pred.shape

        # patch_factor > 1이면 패치를 더 작은 서브패치로 분할
        if self.patch_factor > 1:
            assert H % self.patch_factor == 0 and W % self.patch_factor == 0, (
                f"패치 크기 ({H}, {W})가 patch_factor={self.patch_factor}로 나누어져야 합니다."
            )
            ph = H // self.patch_factor
            pw = W // self.patch_factor
            # (B, H, W) → (B * patch_factor^2, ph, pw)
            pred = (
                pred.view(B, self.patch_factor, ph, self.patch_factor, pw)
                .permute(0, 1, 3, 2, 4)
                .reshape(-1, ph, pw)
            )
            gt = (
                gt.view(B, self.patch_factor, ph, self.patch_factor, pw)
                .permute(0, 1, 3, 2, 4)
                .reshape(-1, ph, pw)
            )

        # 2D FFT 변환
        pred_freq = self.tensor2freq(pred)  # (B', H', W', 2)
        gt_freq = self.tensor2freq(gt)      # (B', H', W', 2)

        # 로그 스케일 (선택): 진폭 범위를 압축하여 학습 안정화
        if self.log_matrix:
            pred_freq = pred_freq.sign() * torch.log1p(pred_freq.abs())
            gt_freq = gt_freq.sign() * torch.log1p(gt_freq.abs())

        # 배치 평균 스펙트럼 (선택): 배치 전체의 대표 주파수 패턴으로 비교
        if self.ave_spectrum:
            pred_freq = pred_freq.mean(dim=0, keepdim=True)
            gt_freq = gt_freq.mean(dim=0, keepdim=True)

        return self._compute_loss(pred_freq, gt_freq)
