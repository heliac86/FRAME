from argparse import Namespace
from math import log2
from typing import Optional, Dict, Any, Union, TYPE_CHECKING, Tuple
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from .hash_grid_torch import HashEmbedder
from ..transform import RigidTransform, ax_transform_points, mat_transform_points
from ..utils import resolution2sigma
# ===== [FF Loss 추가] losses.py에서 FocalFrequencyLoss 및 FF_LOSS 키 가져오기 =====
from .losses import FocalFrequencyLoss, FF_LOSS
# ===== [FF Loss 추가 끝] =====

USE_TORCH = False

if not USE_TORCH:
    try:
        import tinycudann as tcnn
    except:
        logging.warning("Fail to load tinycudann. Will use pytorch implementation.")
        USE_TORCH = True


# key for loss/regularization
D_LOSS = "MSE"
S_LOSS = "logVar"
DS_LOSS = "MSE+logVar"
B_REG = "biasReg"
T_REG = "transReg"
I_REG = "imageReg"
D_REG = "deformReg"
# ===== [Hard Slice Mining] 슬라이스별 MSE 반환용 내부 키 =====
SLICE_MSE_KEY = "_slice_mse_raw"
# ===== [Hard Slice Mining 끝] =====


def build_encoding(**config):
    if USE_TORCH:
        encoding = HashEmbedder(**config)
    else:
        n_input_dims = config.pop("n_input_dims")
        dtype = config.pop("dtype")
        try:
            encoding = tcnn.Encoding(
                n_input_dims=n_input_dims, encoding_config=config, dtype=dtype
            )
        except RuntimeError as e:
            if "TCNN was not compiled with half-precision support" in str(e):
                logging.error(
                    "TCNN was not compiled with half-precision support! "
                    "Try using --single-precision in the nesvor command! "
                )
            raise e
    return encoding


def build_network(**config):
    dtype = config.pop("dtype")
    assert dtype == torch.float16 or dtype == torch.float32
    if dtype == torch.float16 and not USE_TORCH:
        return tcnn.Network(
            n_input_dims=config["n_input_dims"],
            n_output_dims=config["n_output_dims"],
            network_config={
                "otype": "CutlassMLP",
                "activation": config["activation"],
                "output_activation": config["output_activation"],
                "n_neurons": config["n_neurons"],
                "n_hidden_layers": config["n_hidden_layers"],
            },
        )
    else:
        activation = (
            None
            if config["activation"] == "None"
            else getattr(nn, config["activation"])
        )
        output_activation = (
            None
            if config["output_activation"] == "None"
            else getattr(nn, config["output_activation"])
        )
        models = []
        if config["n_hidden_layers"] > 0:
            models.append(nn.Linear(config["n_input_dims"], config["n_neurons"]))
            for _ in range(config["n_hidden_layers"] - 1):
                if activation is not None:
                    models.append(activation())
                models.append(nn.Linear(config["n_neurons"], config["n_neurons"]))
            if activation is not None:
                models.append(activation())
            models.append(nn.Linear(config["n_neurons"], config["n_output_dims"]))
        else:
            models.append(nn.Linear(config["n_input_dims"], config["n_output_dims"]))
        if output_activation is not None:
            models.append(output_activation())
        return nn.Sequential(*models)


def compute_resolution_nlevel(
    bounding_box: torch.Tensor,
    coarsest_resolution: float,
    finest_resolution: float,
    level_scale: float,
    spatial_scaling: float,
) -> Tuple[int, int]:
    base_resolution = (
        (
            (bounding_box[1] - bounding_box[0]).max()
            * spatial_scaling
            / coarsest_resolution
        )
        .ceil()
        .int()
        .item()
    )
    n_levels = (
        (
            torch.log2(
                (bounding_box[1] - bounding_box[0]).max()
                * spatial_scaling
                / finest_resolution
                / base_resolution
            )
            / log2(level_scale)
            + 1
        )
        .ceil()
        .int()
        .item()
    )
    return int(base_resolution), int(n_levels)


# ===== [G6/G7/G14] GatingMLP: 좌표 (+ 선택적 z_repr) → 위치별 level_weights 출력 =====
class GatingMLP(nn.Module):
    """
    공간 적응형 게이팅 네트워크.

    [G6 모드] input_dim=3
        입력: 정규화된 3D 좌표 (N, 3), 범위 [0, 1]^3
    [G7 모드] input_dim=3+n_features_z
        입력: cat([정규화된 3D 좌표, z_repr], dim=-1) (N, 3+n_features_z)
        z_repr: Pass1(ungated)에서 density_net이 출력한 z 피처를 픽셀별로 평균 + detach

    출력: 위치별 level logits (N, n_levels)
          → _apply_gating 내에서 softmax(logits/T) * n_levels 를 거쳐 가중치로 변환
          → clamp(min=min_weight) 적용 (min_weight > 0 이면)

    구조: Linear(input_dim → hidden_dim) → [ReLU|SiLU] → Linear(hidden_dim → n_levels)

    초기화: 마지막 레이어를 zeros 로 초기화
            → 학습 시작 시 모든 위치에서 softmax 출력이 균등 (1.0)
            → 기존 전역 level_weights=zeros 초기 상태와 동일

    [G14 추가 파라미터]
    activation : str
        'relu'  — 기존 방식 (default, G6/G7/G12와 동일)
        'silu'  — SiLU 활성화 (--gating-activation silu)
                  hidden_dim=8처럼 작은 네트워크에서 dying ReLU를 방지.
    min_weight : float
        GatingMLP 출력 weight의 하한 (default=0.0, 하한 없음).
        0.0 초과이면 softmax*n_levels 이후 clamp(min=min_weight) 적용.
        어떤 hash grid level도 완전히 억제되지 않도록 보장.
        예: 0.2 → 모든 level의 최소 가중치 = 0.2
            0.5 → 모든 level의 최소 가중치 = 0.5 (균등에 가깝게 강제)
        주의: clamp 후 weights 합계는 n_levels 보존이 깨지지만,
              GatingMLP가 극단값을 피하는 방향으로 수렴하면
              실제로 clamp가 활성화되는 비율은 낮아짐.
    temperature : float
        softmax temperature T (default=1.0).
        T > 1.0 → 출력이 더 균등해짐 (level 집중 억제)
        T < 1.0 → 출력이 더 날카로워짐 (level 집중 강화, 통상 비권장)
        수식: weights = softmax(logits / T) * n_levels

    [G14 실험 세트]
    G14_clamp02: activation='relu', min_weight=0.2, temperature=1.0  (--gating-min-weight 0.2)
    G14_clamp05: activation='relu', min_weight=0.5, temperature=1.0  (--gating-min-weight 0.5)
    G14_temp15:  activation='relu', min_weight=0.0, temperature=1.5  (--gating-temperature 1.5)
    G14_silu:    activation='silu', min_weight=0.0, temperature=1.0  (--gating-activation silu)
    G12_h8z:     activation='relu', min_weight=0.0, temperature=1.0  (기준, 변경 없음)
    """

    def __init__(
        self,
        n_levels: int,
        hidden_dim: int = 32,
        input_dim: int = 3,
        activation: str = "relu",
        min_weight: float = 0.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.min_weight = min_weight
        self.temperature = temperature

        # ===== [G14] 활성화 함수 선택 =====
        if activation == "silu":
            act_fn = nn.SiLU()
            logging.info("[G14] GatingMLP activation: SiLU")
        else:
            act_fn = nn.ReLU()
            logging.info("[G14] GatingMLP activation: ReLU (default)")
        # ===== [G14 끝] =====

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, n_levels),
        )
        # 마지막 레이어 zeros 초기화 → 초기 출력 = 균등 가중치
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        logging.info(
            "[G6/G7/G14] GatingMLP initialized: %d -> %d -> %d "
            "| activation=%s | min_weight=%.2f | temperature=%.2f | params=%d",
            input_dim,
            hidden_dim,
            n_levels,
            activation,
            min_weight,
            temperature,
            sum(p.numel() for p in self.parameters()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, input_dim)
           G6: (N, 3)           — 정규화된 픽셀 대표 좌표
           G7: (N, 3+n_feat_z)  — cat([좌표, z_repr])
        반환: (N, n_levels) logit
        """
        return self.net(x)
# ===== [G6/G7/G14 끝] =====


class INR(nn.Module):
    def __init__(
        self, bounding_box: torch.Tensor, args: Namespace, spatial_scaling: float = 1.0
    ) -> None:
        super().__init__()
        if TYPE_CHECKING:
            self.bounding_box: torch.Tensor
        self.register_buffer("bounding_box", bounding_box)
        # hash grid encoding
        base_resolution, n_levels = compute_resolution_nlevel(
            self.bounding_box,
            args.coarsest_resolution,
            args.finest_resolution,
            args.level_scale,
            spatial_scaling,
        )

        self.n_levels = n_levels
        self.n_features_per_level = args.n_features_per_level
        self.n_features_z = args.n_features_z

        # ===== [k_norm] 역정규화 스케일 인자 =====
        self.register_buffer("v_mean", torch.tensor(1.0, dtype=torch.float32))
        # ===== [k_norm 끝] =====

        self.encoding = build_encoding(
            n_input_dims=3,
            otype="HashGrid",
            n_levels=n_levels,
            n_features_per_level=args.n_features_per_level,
            log2_hashmap_size=args.log2_hashmap_size,
            base_resolution=base_resolution,
            per_level_scale=args.level_scale,
            dtype=args.dtype,
        )
        # density net
        self.density_net = build_network(
            n_input_dims=n_levels * args.n_features_per_level,
            n_output_dims=1 + args.n_features_z,
            activation="ReLU",
            output_activation="None",
            n_neurons=args.width,
            n_hidden_layers=args.depth,
            dtype=torch.float32 if args.img_reg_autodiff else args.dtype,
        )

        # ===== [G1] Softmax-normalized Gating =====
        self.use_gating = not getattr(args, "no_gating", False)
        self.gating_mode = getattr(args, "gating_mode", "standard")

        # ===== [G6] 공간 적응형 게이팅 여부 =====
        self.spatial_gating = (
            self.use_gating and getattr(args, "spatial_gating", False)
        )
        # ===== [G6 끝] =====

        # ===== [G7] z_repr 기반 게이팅 여부 =====
        # spatial_gating=True 일 때만 유효. --no-z-gating 플래그로 비활성화 가능.
        self.z_gating = (
            self.spatial_gating and not getattr(args, "no_z_gating", False)
        )
        # ===== [G7 끝] =====

        if self.use_gating:
            if self.spatial_gating:
                # [G6/G7/G14] GatingMLP: 좌표 (+ 선택적 z_repr) → 위치별 level logits
                gating_hidden_dim = getattr(args, "gating_hidden_dim", 32)
                # [G7] z_gating=True면 입력 차원 확장: 3 → 3 + n_features_z
                gating_input_dim = (
                    3 + args.n_features_z if self.z_gating else 3
                )
                # ===== [G14] 새 파라미터 읽기 =====
                gating_activation  = getattr(args, "gating_activation",  "relu")
                gating_min_weight  = getattr(args, "gating_min_weight",  0.0)
                gating_temperature = getattr(args, "gating_temperature", 1.0)
                # ===== [G14 끝] =====
                self.gating_net = GatingMLP(
                    n_levels=n_levels,
                    hidden_dim=gating_hidden_dim,
                    input_dim=gating_input_dim,
                    activation=gating_activation,
                    min_weight=gating_min_weight,
                    temperature=gating_temperature,
                )
                logging.info(
                    "[G6/G7/G14] Spatial Gating enabled: n_levels=%d, hidden_dim=%d, "
                    "input_dim=%d (z_gating=%s) | activation=%s min_weight=%.2f temperature=%.2f",
                    n_levels,
                    gating_hidden_dim,
                    gating_input_dim,
                    self.z_gating,
                    gating_activation,
                    gating_min_weight,
                    gating_temperature,
                )
            else:
                # [G1] 전역 level_weights (기존 방식)
                self.level_weights = nn.Parameter(
                    torch.zeros(n_levels, dtype=torch.float32)
                )
                logging.info(
                    "[G1] Hash grid Gating enabled: n_levels=%d, mode=%s, "
                    "init=zeros (softmax -> uniform 1.0), normalization=softmax*n_levels",
                    n_levels,
                    self.gating_mode,
                )
        # ===== [G1 끝] =====

        # logging
        logging.debug(
            "hyperparameters for hash grid encoding: "
            + "lowest_grid_size=%d, highest_grid_size=%d, scale=%1.2f, n_levels=%d",
            base_resolution,
            int(base_resolution * args.level_scale ** (n_levels - 1)),
            args.level_scale,
            n_levels,
        )
        logging.debug(
            "bounding box for reconstruction (mm): "
            + "x=[%f, %f], y=[%f, %f], z=[%f, %f]",
            self.bounding_box[0, 0],
            self.bounding_box[1, 0],
            self.bounding_box[0, 1],
            self.bounding_box[1, 1],
            self.bounding_box[0, 2],
            self.bounding_box[1, 2],
        )

    # ===== [G6/G7] 공통 헬퍼: x_repr 및 n_samples 추출 =====
    def _get_x_repr_and_n_samples(
        self, x_norm: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        x_norm이 (N, n_samples, 3)이면 PSF 샘플 축을 평균내어 (N, 3) 반환.
        x_norm이 이미 (N, 3)이면 그대로 반환.
        반환: (x_repr: FloatTensor (N, 3), n_samples: int)
        """
        if x_norm.dim() == 3:
            # 학습 경로: (N, n_samples, 3) → (N, 3)
            return x_norm.mean(dim=1).float(), x_norm.shape[1]
        else:
            # 추론 경로 또는 이미 flat: (N, 3)
            return x_norm.float(), 1
    # ===== [헬퍼 끝] =====

    def _apply_gating(
        self,
        pe: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        n_samples: int = 1,
        z_repr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """레벨별 softmax-normalized 가중치를 pe에 적용.

        [G1 전역 모드]
            pe shape : (N, n_levels * n_features_per_level)  — N = 픽셀*n_samples
            n_samples: 사용 안 함 (전역 weights는 확장 불필요)
            반환 weights shape : (n_levels,)

        [G6 공간 적응형 모드] z_repr=None
            x shape  : (N_px, 3) 픽셀 대표 정규화 좌표 (PSF 샘플 평균), 필수
            pe shape : (N_px * n_samples, n_levels * n_features_per_level)
            n_samples: PSF 샘플 수. weights (N_px, n_levels) →
                       repeat_interleave(n_samples) → (N_px*n_samples, n_levels)
            반환 weights shape : (N_px, n_levels)

        [G7 공간+z 모드] z_repr is not None
            x shape    : (N_px, 3)            — 픽셀 대표 좌표
            z_repr shape: (N_px, n_features_z) — Pass1 density_net 출력 z 피처, detach됨
            GatingMLP 입력: cat([x, z_repr], dim=-1) (N_px, 3+n_features_z)

        [G14 수정]
            temperature: logit을 temperature로 나눈 뒤 softmax
                         gating_net.temperature 에서 자동 참조
            min_weight:  softmax*n_levels 이후 clamp(min=min_weight) 적용
                         gating_net.min_weight 에서 자동 참조
                         min_weight=0.0 이면 clamp 미적용 (G12 동작과 동일)
        """
        if self.spatial_gating:
            # [G6/G7] 픽셀 대표 입력 준비
            assert x is not None, "[G6/G7] spatial_gating=True 이지만 x(좌표)가 전달되지 않았습니다."

            if z_repr is not None:
                # [G7] 좌표 + z 피처 결합
                gating_input = torch.cat([x.float(), z_repr.float()], dim=-1)  # (N_px, 3+n_feat_z)
                logging.debug("[G7] gating_input shape: %s", list(gating_input.shape))
            else:
                # [G6] 좌표만 사용
                gating_input = x.float()  # (N_px, 3)

            logits = self.gating_net(gating_input)                 # (N_px, n_levels)

            # ===== [G14] temperature scaling =====
            T = self.gating_net.temperature
            if T != 1.0:
                logits = logits / T
            # ===== [G14 끝] =====

            weights = F.softmax(logits, dim=-1) * self.n_levels    # (N_px, n_levels)

            # ===== [G14] min_weight clamp =====
            min_w = self.gating_net.min_weight
            if min_w > 0.0:
                weights = weights.clamp(min=min_w)
            # ===== [G14 끝] =====

            if n_samples > 1:
                # weights를 n_samples 만큼 반복 확장: (N_px, n_levels) → (N_px*n_samples, n_levels)
                weights_expanded = weights.repeat_interleave(n_samples, dim=0)
            else:
                weights_expanded = weights

            # pe: (N_px*n_samples, n_levels*n_feat)
            pe_gate = pe.view(-1, self.n_levels, self.n_features_per_level)
            pe_gate = pe_gate * weights_expanded.unsqueeze(-1)     # (N_px*n_samples, n_levels, n_feat)
            logging.debug(
                "[G6/G7/G14] spatial weights mean per level: %s",
                [f"{v:.4f}" for v in weights.detach().mean(0).cpu().tolist()],
            )
        else:
            # [G1] 전역 level_weights
            weights = F.softmax(self.level_weights, dim=0) * self.n_levels
            pe_gate = pe.view(-1, self.n_levels, self.n_features_per_level)
            pe_gate = pe_gate * weights.view(1, self.n_levels, 1)
            logging.debug(
                "[G1] level_weights (softmax): %s",
                [f"{v:.4f}" for v in weights.detach().cpu().tolist()],
            )
        return pe_gate.view(-1, self.n_levels * self.n_features_per_level), weights

    def forward(self, x: torch.Tensor):
        # x shape: (N, n_samples, 3) 또는 (N, 3)
        x_norm = (x - self.bounding_box[0]) / (self.bounding_box[1] - self.bounding_box[0])
        prefix_shape = x_norm.shape[:-1]
        x_flat = x_norm.view(-1, x_norm.shape[-1])    # (N*n_samples, 3) 또는 (N, 3)
        pe = self.encoding(x_flat)
        if not self.training:
            pe = pe.to(dtype=x_flat.dtype)

        # ===== [G1/G6/G7] 게이팅 적용 =====
        gating_weights = None
        if self.use_gating:
            if self.spatial_gating:
                # [G6/G7] 픽셀 대표 좌표 추출 (헬퍼 사용)
                x_repr, n_samples = self._get_x_repr_and_n_samples(x_norm)

                if self.z_gating:
                    # ===== [G7] 2-pass: Pass1 ungated → z_repr 획득 =====
                    # Pass1: gating 없이 density_net을 한 번 통과 → z 피처 추출
                    z1 = self.density_net(pe)                                   # (N*n_samples, 1+n_feat_z)
                    z_feat1 = z1[:, 1:]                                         # (N*n_samples, n_feat_z)
                    # 픽셀별 대표 z: n_samples 축 평균
                    if n_samples > 1:
                        z_feat1 = z_feat1.view(-1, n_samples, self.n_features_z).mean(dim=1)  # (N, n_feat_z)
                    z_repr = z_feat1.detach()                                   # gradient 차단
                    # Pass2: z_repr을 포함한 gating → pe_gated
                    pe, gating_weights = self._apply_gating(
                        pe, x_repr, n_samples=n_samples, z_repr=z_repr
                    )
                    # ===== [G7 2-pass 끝] =====
                else:
                    # [G6] 좌표만으로 gating
                    pe, gating_weights = self._apply_gating(pe, x_repr, n_samples=n_samples)
            else:
                # [G1] 전역 gating: x 불필요
                pe, gating_weights = self._apply_gating(pe)
        # ===== [G1/G6/G7 끝] =====

        z = self.density_net(pe)
        density = F.softplus(z[..., 0].view(prefix_shape))
        if self.training:
            return density, pe, z, gating_weights
        else:
            return density, gating_weights

    def forward_ff_direct(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """ff_direct 모드 전용 forward.
        encoding은 detach → FF Loss gradient가 오직 gating 파라미터로만 흐름.
        patch_forward에서만 호출됨.
        """
        # x shape: (N, n_samples, 3) 또는 (N, 3)
        x_norm = (x - self.bounding_box[0]) / (self.bounding_box[1] - self.bounding_box[0])
        prefix_shape = x_norm.shape[:-1]
        x_flat = x_norm.view(-1, x_norm.shape[-1])
        # ===== [G1_D] encoding detach =====
        pe = self.encoding(x_flat).detach()
        if not self.training:
            pe = pe.to(dtype=x_flat.dtype)
        gating_weights = None
        if self.use_gating:
            if self.spatial_gating:
                # [G6/G7] 픽셀 대표 좌표 추출 (헬퍼 사용)
                x_repr, n_samples = self._get_x_repr_and_n_samples(x_norm)

                if self.z_gating:
                    # [G7] 2-pass (ff_direct 모드: pe는 이미 detach됨)
                    z1 = self.density_net(pe)
                    z_feat1 = z1[:, 1:]
                    if n_samples > 1:
                        z_feat1 = z_feat1.view(-1, n_samples, self.n_features_z).mean(dim=1)
                    z_repr = z_feat1.detach()
                    pe, gating_weights = self._apply_gating(
                        pe, x_repr, n_samples=n_samples, z_repr=z_repr
                    )
                else:
                    pe, gating_weights = self._apply_gating(pe, x_repr, n_samples=n_samples)
            else:
                pe, gating_weights = self._apply_gating(pe)
        # ===== [G1_D 끝] =====
        z = self.density_net(pe)
        density = F.softplus(z[..., 0].view(prefix_shape))
        return density, pe, z, gating_weights

    def sample_batch(
        self,
        xyz: torch.Tensor,
        transformation: Optional[RigidTransform],
        psf_sigma: Union[float, torch.Tensor],
        n_samples: int,
    ) -> torch.Tensor:
        if n_samples > 1:
            if isinstance(psf_sigma, torch.Tensor):
                psf_sigma = psf_sigma.view(-1, 1, 3)
            xyz_psf = torch.randn(
                xyz.shape[0], n_samples, 3, dtype=xyz.dtype, device=xyz.device
            )
            xyz = xyz[:, None] + xyz_psf * psf_sigma
        else:
            xyz = xyz[:, None]
        if transformation is not None:
            trans_first = transformation.trans_first
            mat = transformation.matrix(trans_first)
            xyz = mat_transform_points(mat[:, None], xyz, trans_first)
        return xyz


class DeformNet(nn.Module):
    def __init__(
        self, bounding_box: torch.Tensor, args: Namespace, spatial_scaling: float = 1.0
    ) -> None:
        super().__init__()
        if TYPE_CHECKING:
            self.bounding_box: torch.Tensor
        self.register_buffer("bounding_box", bounding_box)
        base_resolution, n_levels = compute_resolution_nlevel(
            bounding_box,
            args.coarsest_resolution_deform,
            args.finest_resolution_deform,
            args.level_scale_deform,
            spatial_scaling,
        )
        level_scale = args.level_scale_deform

        self.encoding = build_encoding(
            n_input_dims=3,
            otype="HashGrid",
            n_levels=n_levels,
            n_features_per_level=args.n_features_per_level_deform,
            log2_hashmap_size=args.log2_hashmap_size,
            base_resolution=base_resolution,
            per_level_scale=level_scale,
            dtype=args.dtype,
            interpolation="Smoothstep",
        )
        self.deform_net = build_network(
            n_input_dims=n_levels * args.n_features_per_level_deform
            + args.n_features_deform,
            n_output_dims=3,
            activation="Tanh",
            output_activation="None",
            n_neurons=args.width,
            n_hidden_layers=2,
            dtype=torch.float32,
        )
        for p in self.deform_net.parameters():
            torch.nn.init.uniform_(p, a=-1e-4, b=1e-4)
        logging.debug(
            "hyperparameters for hash grid encoding (deform net): "
            + "lowest_grid_size=%d, highest_grid_size=%d, scale=%1.2f, n_levels=%d",
            base_resolution,
            int(base_resolution * level_scale ** (n_levels - 1)),
            level_scale,
            n_levels,
        )

    def forward(self, x: torch.Tensor, e: torch.Tensor):
        x = (x - self.bounding_box[0]) / (self.bounding_box[1] - self.bounding_box[0])
        x_shape = x.shape
        x = x.view(-1, x.shape[-1])
        pe = self.encoding(x)
        inputs = torch.cat((pe, e.reshape(-1, e.shape[-1])), -1)
        outputs = self.deform_net(inputs) + x
        outputs = (
            outputs * (self.bounding_box[1] - self.bounding_box[0])
            + self.bounding_box[0]
        )
        return outputs.view(x_shape)


class NeSVoR(nn.Module):
    def __init__(
        self,
        transformation: RigidTransform,
        resolution: torch.Tensor,
        v_mean: float,
        bounding_box: torch.Tensor,
        spatial_scaling: float,
        args: Namespace,
    ) -> None:
        super().__init__()
        if "cpu" in str(args.device):  # CPU mode
            global USE_TORCH
            USE_TORCH = True
        else:
            torch.cuda.set_device(args.device)
        self.spatial_scaling = spatial_scaling
        self.args = args
        self.n_slices = 0
        self.trans_first = True
        self.transformation = transformation
        self.psf_sigma = resolution2sigma(resolution, isotropic=False)
        self.delta = args.delta * v_mean
        self.build_network(bounding_box)
        self.to(args.device)

    @property
    def transformation(self) -> RigidTransform:
        return RigidTransform(self.axisangle.detach(), self.trans_first)

    @transformation.setter
    def transformation(self, value: RigidTransform) -> None:
        if self.n_slices == 0:
            self.n_slices = len(value)
        else:
            assert self.n_slices == len(value)
        axisangle = value.axisangle(self.trans_first)
        if TYPE_CHECKING:
            self.axisangle_init: torch.Tensor
        self.register_buffer("axisangle_init", axisangle.detach().clone())
        if not self.args.no_transformation_optimization:
            self.axisangle = nn.Parameter(axisangle.detach().clone())
        else:
            self.register_buffer("axisangle", axisangle.detach().clone())

    def build_network(self, bounding_box) -> None:
        if self.args.n_features_slice:
            self.slice_embedding = nn.Embedding(
                self.n_slices, self.args.n_features_slice
            )
        if not self.args.no_slice_scale:
            self.logit_coef = nn.Parameter(
                torch.zeros(self.n_slices, dtype=torch.float32)
            )
        if not self.args.no_slice_variance:
            self.log_var_slice = nn.Parameter(
                torch.zeros(self.n_slices, dtype=torch.float32)
            )
        if self.args.deformable:
            self.deform_embedding = nn.Embedding(
                self.n_slices, self.args.n_features_deform
            )
            self.deform_net = DeformNet(bounding_box, self.args, self.spatial_scaling)
        # INR
        self.inr = INR(bounding_box, self.args, self.spatial_scaling)
        # sigma net
        if not self.args.no_pixel_variance:
            self.sigma_net = build_network(
                n_input_dims=self.args.n_features_slice + self.args.n_features_z,
                n_output_dims=1,
                activation="ReLU",
                output_activation="None",
                n_neurons=self.args.width,
                n_hidden_layers=self.args.depth,
                dtype=self.args.dtype,
            )
        # bias net
        if self.args.n_levels_bias:
            self.b_net = build_network(
                n_input_dims=self.args.n_levels_bias * self.args.n_features_per_level
                + self.args.n_features_slice,
                n_output_dims=1,
                activation="ReLU",
                output_activation="None",
                n_neurons=self.args.width,
                n_hidden_layers=self.args.depth,
                dtype=self.args.dtype,
            )
        # ===== [FF Loss 추가] FocalFrequencyLoss 인스턴스 초기화 =====
        if getattr(self.args, "weight_ff_loss", 0.0) > 0:
            self.ff_loss_fn = FocalFrequencyLoss(
                alpha=getattr(self.args, "ff_alpha", 1.0),
            )
        else:
            self.ff_loss_fn = None
        # ===== [FF Loss 추가 끝] =====

    def forward(
        self,
        xyz: torch.Tensor,
        v: torch.Tensor,
        slice_idx: torch.Tensor,
    ) -> Dict[str, Any]:
        batch_size = xyz.shape[0]
        n_samples = self.args.n_samples
        xyz_psf = torch.randn(
            batch_size, n_samples, 3, dtype=xyz.dtype, device=xyz.device
        )
        psf_sigma = self.psf_sigma[slice_idx][:, None]
        t = self.axisangle[slice_idx][:, None]
        xyz = ax_transform_points(
            t, xyz[:, None] + xyz_psf * psf_sigma, self.trans_first
        )

        xyz_ori = xyz
        if self.args.deformable:
            de = self.deform_embedding(slice_idx)[:, None].expand(-1, n_samples, -1)
            xyz = self.deform_net(xyz, de)

        if self.args.n_features_slice:
            se = self.slice_embedding(slice_idx)[:, None].expand(-1, n_samples, -1)
        else:
            se = None
        results = self.net_forward(xyz, se)
        density = results["density"]
        if "log_bias" in results:
            log_bias = results["log_bias"]
            bias = log_bias.exp()
            bias_detach = bias.detach()
        else:
            log_bias = 0
            bias = 1
            bias_detach = 1
        if "log_var" in results:
            log_var = results["log_var"]
            var = log_var.exp()
        else:
            log_var = 0
            var = 1
        if not self.args.no_slice_scale:
            c: Any = F.softmax(self.logit_coef, 0)[slice_idx] * self.n_slices
        else:
            c = 1
        v_out = (bias * density).mean(-1)
        v_out = c * v_out
        if not self.args.no_pixel_variance:
            var = (bias_detach * var).mean(-1)
            c_detach = c.detach() if isinstance(c, torch.Tensor) else c
            var = c_detach * var
            var = var**2
        if not self.args.no_slice_variance:
            var = var + self.log_var_slice.exp()[slice_idx]
        pixel_mse = (v_out - v) ** 2 / (2 * var)
        losses = {D_LOSS: pixel_mse.mean()}
        if not (self.args.no_pixel_variance and self.args.no_slice_variance):
            losses[S_LOSS] = 0.5 * var.log().mean()
            losses[DS_LOSS] = losses[D_LOSS] + losses[S_LOSS]
        if not self.args.no_transformation_optimization:
            losses[T_REG] = self.trans_loss(trans_first=self.trans_first)
        if self.args.n_levels_bias:
            losses[B_REG] = log_bias.mean() ** 2
        if self.args.deformable:
            losses[D_REG] = self.deform_reg(xyz, xyz_ori, de)
        losses[I_REG] = self.img_reg(density, xyz)
        losses[SLICE_MSE_KEY] = (pixel_mse.detach().float(), slice_idx)
        return losses

    def patch_forward(
        self,
        xyz_patch: torch.Tensor,
        v_patch: torch.Tensor,
        slice_idx_patch: torch.Tensor,
        valid_mask_patch: torch.Tensor,
    ) -> Dict[str, Any]:
        n = xyz_patch.shape[0]
        PP = xyz_patch.shape[1]
        P = int(PP ** 0.5)

        xyz_flat = xyz_patch.view(n * PP, 3) / self.spatial_scaling
        slice_idx_flat = slice_idx_patch.repeat_interleave(PP)

        n_samples = self.args.n_samples
        xyz_psf = torch.randn(
            n * PP, n_samples, 3, dtype=xyz_flat.dtype, device=xyz_flat.device
        )
        psf_sigma = self.psf_sigma[slice_idx_flat][:, None]
        t = self.axisangle[slice_idx_flat][:, None]
        xyz_t = ax_transform_points(
            t, xyz_flat[:, None] + xyz_psf * psf_sigma, self.trans_first
        )

        if self.args.deformable:
            de = self.deform_embedding(slice_idx_flat)[:, None].expand(
                -1, n_samples, -1
            )
            xyz_t = self.deform_net(xyz_t, de)

        if self.args.n_features_slice:
            se = self.slice_embedding(slice_idx_flat)[:, None].expand(
                -1, n_samples, -1
            )
        else:
            se = None

        if self.inr.use_gating and self.inr.gating_mode == "ff_direct":
            results = self.net_forward_ff_direct(xyz_t, se)
        else:
            results = self.net_forward(xyz_t, se)

        density = results["density"]

        if "log_bias" in results:
            bias = results["log_bias"].exp()
        else:
            bias = 1

        if not self.args.no_slice_scale:
            c: Any = F.softmax(self.logit_coef, 0)[slice_idx_flat] * self.n_slices
        else:
            c = 1

        v_pred = (bias * density).mean(-1)
        v_pred = c * v_pred
        v_pred_patch = v_pred.view(n, P, P)

        mask_float = valid_mask_patch.float()
        bg_mask = 1.0 - mask_float
        valid_sum_pred = (v_pred_patch * mask_float).sum(dim=(-2, -1))
        valid_sum_gt   = (v_patch      * mask_float).sum(dim=(-2, -1))
        valid_count    = mask_float.sum(dim=(-2, -1)).clamp(min=1.0)
        mean_pred = valid_sum_pred / valid_count
        mean_gt   = valid_sum_gt   / valid_count
        v_pred_filled = v_pred_patch * mask_float + mean_pred.view(n, 1, 1) * bg_mask
        v_gt_filled   = v_patch      * mask_float + mean_gt.view(n, 1, 1)   * bg_mask

        ff_loss = self.ff_loss_fn(v_pred_filled, v_gt_filled)
        return {FF_LOSS: ff_loss}

    def net_forward(
        self,
        x: torch.Tensor,
        se: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        density, pe, z, _gating_weights = self.inr(x)
        prefix_shape = density.shape
        results = {"density": density}

        zs = []
        if se is not None:
            zs.append(se.reshape(-1, se.shape[-1]))

        if self.args.n_levels_bias:
            pe_bias = pe[
                ..., : self.args.n_levels_bias * self.args.n_features_per_level
            ]
            results["log_bias"] = self.b_net(torch.cat(zs + [pe_bias], -1)).view(
                prefix_shape
            )

        if not self.args.no_pixel_variance:
            zs.append(z[..., 1:])
            results["log_var"] = self.sigma_net(torch.cat(zs, -1)).view(prefix_shape)

        return results

    def net_forward_ff_direct(
        self,
        x: torch.Tensor,
        se: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """ff_direct 모드 전용: encoding detach forward 사용."""
        density, pe, z, _gating_weights = self.inr.forward_ff_direct(x)
        prefix_shape = density.shape
        results = {"density": density}

        zs = []
        if se is not None:
            zs.append(se.reshape(-1, se.shape[-1]))

        if self.args.n_levels_bias:
            pe_bias = pe[
                ..., : self.args.n_levels_bias * self.args.n_features_per_level
            ]
            results["log_bias"] = self.b_net(torch.cat(zs + [pe_bias], -1)).view(
                prefix_shape
            )

        if not self.args.no_pixel_variance:
            zs.append(z[..., 1:])
            results["log_var"] = self.sigma_net(torch.cat(zs, -1)).view(prefix_shape)

        return results

    def trans_loss(self, trans_first: bool = True) -> torch.Tensor:
        x = RigidTransform(self.axisangle, trans_first=trans_first)
        y = RigidTransform(self.axisangle_init, trans_first=trans_first)
        err = y.inv().compose(x).axisangle(trans_first=trans_first)
        loss_R = torch.mean(err[:, :3] ** 2)
        loss_T = torch.mean(err[:, 3:] ** 2)
        return loss_R + 1e-3 * self.spatial_scaling * self.spatial_scaling * loss_T

    def img_reg(self, density, xyz):
        if self.args.image_regularization == "none":
            return torch.zeros((1,), dtype=density.dtype, device=density.device)

        if self.args.img_reg_autodiff:
            n_sample = 4
            xyz = xyz[:, :n_sample].flatten(0, 1).detach()
            xyz.requires_grad_()
            density, _ = self.inr(xyz)
            grad = (
                torch.autograd.grad((density.sum(),), (xyz,), create_graph=True)[0]
                / self.spatial_scaling
            )
            grad2 = grad.pow(2)
        else:
            xyz = xyz * self.spatial_scaling
            d_density = density - torch.flip(density, (1,))
            dx2 = ((xyz - torch.flip(xyz, (1,))) ** 2).sum(-1) + 1e-6
            grad2 = d_density**2 / dx2

        if self.args.image_regularization == "TV":
            return grad2.sqrt().mean()
        elif self.args.image_regularization == "edge":
            return self.delta * (
                (1 + grad2 / (self.delta * self.delta)).sqrt().mean() - 1
            )
        elif self.args.image_regularization == "L2":
            return grad2.mean()
        else:
            raise ValueError("unknown image regularization!")

    def deform_reg(self, out, xyz, e):
        if True:
            n_sample = 4
            x = xyz[:, :n_sample].flatten(0, 1).detach()
            e = e[:, :n_sample].flatten(0, 1).detach()
            x.requires_grad_()
            outputs = self.deform_net(x, e)
            grads = []
            out_sum = []
            for i in range(3):
                out_sum.append(outputs[:, i].sum())
                grads.append(
                    torch.autograd.grad((out_sum[-1],), (x,), create_graph=True)[0]
                )
            jacobian = torch.stack(grads, -1)
            jtj = torch.matmul(jacobian, jacobian.transpose(-1, -2))
            I = torch.eye(3, dtype=jacobian.dtype, device=jacobian.device).unsqueeze(0)
            sq_residual = ((jtj - I) ** 2).sum((-2, -1))
            return torch.nan_to_num(sq_residual, 0.0, 0.0, 0.0).mean()
        else:
            out = out - xyz
            d_out2 = ((out - torch.flip(out, (1,))) ** 2).sum(-1) + 1e-6
            dx2 = ((xyz - torch.flip(xyz, (1,))) ** 2).sum(-1) + 1e-6
            dd_dx = d_out2.sqrt() / dx2.sqrt()
            return F.smooth_l1_loss(dd_dx, torch.zeros_like(dd_dx).detach(), beta=1e-3)
