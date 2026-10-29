from argparse import Namespace
from typing import List, Tuple
import time
import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
from ..utils import MovingAverage, log_params, TrainLogger
# ===== [FF Loss 추가] 모델 임포트에 FF_LOSS 키 추가 =====
from .models import INR, NeSVoR, D_LOSS, S_LOSS, DS_LOSS, I_REG, B_REG, T_REG, D_REG, FF_LOSS, SLICE_MSE_KEY
# ===== [FF Loss 추가 끝] =====
from ..transform import RigidTransform
from ..image import Volume, Slice
from .data import PointDataset

# ===== [G2] Diversity Loss 키 =====
DIVERSITY_LOSS = "diversity_loss"
# ===== [G2 끝] =====


def train(slices: List[Slice], args: Namespace) -> Tuple[INR, List[Slice], Volume]:
    # create training dataset
    dataset = PointDataset(slices)
    if args.n_epochs is not None:
        args.n_iter = args.n_epochs * (dataset.v.numel() // args.batch_size)

    use_scaling = True
    use_centering = True
    # perform centering and scaling
    spatial_scaling = 30.0 if use_scaling else 1
    bb = dataset.bounding_box
    center = (bb[0] + bb[-1]) / 2 if use_centering else torch.zeros_like(bb[0])
    ax = (
        RigidTransform(torch.cat([torch.zeros_like(center), -center])[None])
        .compose(dataset.transformation)
        .axisangle()
    )
    ax[:, -3:] /= spatial_scaling
    transformation = RigidTransform(ax)
    dataset.xyz /= spatial_scaling

    model = NeSVoR(
        transformation,
        dataset.resolution / spatial_scaling,
        dataset.mean,
        (bb - center) / spatial_scaling,
        spatial_scaling,
        args,
    )

    # ===== [k_norm] 훈련 타겟 정규화 =====
    v_mean = dataset.mean
    logging.info("[k_norm] Training target normalization: v_mean = %.4f", v_mean)
    dataset.normalize(v_mean)
    model.inr.v_mean.fill_(v_mean)
    model.delta = model.delta / v_mean
    logging.info("[k_norm] model.delta rescaled to %.6f", model.delta)
    # ===== [k_norm 끝] =====

    # setup optimizer
    params_net = []
    params_encoding = []
    # ===== [G1_hlr / G6] gating 파라미터 그룹 분리 =====
    # level_weights (G1~G5 전역 gating) 및 gating_net (G6 GatingMLP) 모두 포함
    params_gating = []
    gating_lr_scale = getattr(args, "gating_lr_scale", 1.0)
    # ===== [G1_hlr 끝] =====
    for name, param in model.named_parameters():
        if param.numel() > 0:
            # ===== [G1_hlr / G6] level_weights 및 gating_net 파라미터를 별도 그룹으로 분리 =====
            if "level_weights" in name or "gating_net" in name:
                params_gating.append(param)
            # ===== [G1_hlr / G6 끝] =====
            elif "_net" in name:
                params_net.append(param)
            else:
                params_encoding.append(param)
    logging.debug(log_params(model))

    # ===== [G1_hlr] optimizer 그룹 구성: gating 그룹 추가 =====
    param_groups = [
        {"name": "encoding", "params": params_encoding},
        {"name": "net",      "params": params_net,     "weight_decay": 1e-2},
    ]
    if params_gating:
        param_groups.append({
            "name": "gating",
            "params": params_gating,
            "lr": args.learning_rate * gating_lr_scale,
        })
        logging.info(
            "[G1_hlr/G6] Gating optimizer group: lr=%.2e (base=%.2e x scale=%.1f), n_params=%d",
            args.learning_rate * gating_lr_scale,
            args.learning_rate,
            gating_lr_scale,
            sum(p.numel() for p in params_gating),
        )
    # ===== [G1_hlr 끝] =====

    optimizer = torch.optim.AdamW(
        params=param_groups,
        lr=args.learning_rate,
        betas=(0.9, 0.99),
        eps=1e-15,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=list(range(1, len(args.milestones) + 1)),
        gamma=args.gamma,
    )
    decay_milestones = [int(m * args.n_iter) for m in args.milestones]
    fp16 = not args.single_precision
    scaler = torch.cuda.amp.GradScaler(
        init_scale=1.0,
        enabled=fp16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
    )
    model.train()
    loss_weights = {
        D_LOSS: 1,
        S_LOSS: 1,
        T_REG: args.weight_transformation,
        B_REG: args.weight_bias,
        I_REG: args.weight_image,
        D_REG: args.weight_deform,
        FF_LOSS: getattr(args, "weight_ff_loss", 0.0),
        # ===== [G2] Diversity Loss 가중치 =====
        DIVERSITY_LOSS: getattr(args, "weight_diversity_loss", 0.0),
        # ===== [G2 끝] =====
    }

    # ===== [G6] spatial_gating 여부 확인 =====
    spatial_gating = getattr(model.inr, "spatial_gating", False)
    # ===== [G6 끝] =====

    # ===== [G3] Diversity Loss 설정 =====
    use_diversity_loss = (
        not getattr(args, "no_gating", False)
        and loss_weights[DIVERSITY_LOSS] > 0.0
        and bool(params_gating)
        and not spatial_gating  # [G6] GatingMLP 사용 시 전역 diversity loss 비활성화
    )
    # diversity loss 가 적용될 공간: 'raw'(logit) 또는 'softmax'(실제 gating 출력)
    diversity_loss_space = getattr(args, "diversity_loss_space", "raw")
    # target_var: 이 값 이상으로 업되면 loss=0 (발산 방지 브레이크)
    target_diversity_var = getattr(args, "target_diversity_var", 0.05)
    # gating grad clip (0 = 비활성화)
    gating_grad_clip = getattr(args, "gating_grad_clip", 0.0)
    # ===== [B] diversity_loss_fn: 'variance' | 'entropy' | 'gini' =====
    # variance (default, G5_lowvar 호환): relu(target - var)  → var을 높이는 방향
    # entropy:                             relu(entropy - target) → entropy를 낮추는 방향 (분화)
    # gini:                                relu(gini - target)    → gini를 낮추는 방향 (분화)
    diversity_loss_fn = getattr(args, "diversity_loss_fn", "variance")
    # ===== [B 끝] =====

    if spatial_gating and loss_weights[DIVERSITY_LOSS] > 0.0:
        logging.info(
            "[G6] --weight-diversity-loss is set but --spatial-gating is enabled. "
            "Global diversity loss is skipped for GatingMLP (per-coordinate gating). "
            "Use spatial regularization instead if needed."
        )

    # ===== [안전성 개선] diversity loss 파라미터: params_gating[0] 대신 model.inr.level_weights 직접 참조 =====
    # params_gating[0]은 파라미터 등록 순서에 의존하므로, 나중에 코드가 변경되면
    # 의도치 않은 파라미터를 참조할 수 있음. model.inr.level_weights를 직접 사용하면
    # 항상 올바른 파라미터를 참조하며, 문제 발생 시 즉시 명시적 에러가 발생.
    if use_diversity_loss:
        _level_weights_param = model.inr.level_weights
        assert isinstance(_level_weights_param, torch.nn.Parameter), (
            "[Diversity Loss] model.inr.level_weights가 nn.Parameter가 아닙니다. "
            "--no-gating 여부 또는 --spatial-gating 설정을 확인하세요."
        )
        logging.info(
            "[G3/B] Diversity Loss enabled: weight=%.4f, space=%s, fn=%s, target_var=%.4f, grad_clip=%.2f",
            loss_weights[DIVERSITY_LOSS],
            diversity_loss_space,
            diversity_loss_fn,
            target_diversity_var,
            gating_grad_clip,
        )
    # ===== [안전성 개선 끝] =====

    # ===== [G3_warmup] Density net warmup freeze 설정 =====
    gating_warmup_iters = 0
    if not getattr(args, "no_gating", False) and bool(params_gating):
        gating_warmup_iters = getattr(args, "gating_warmup_iters", 0)
    if gating_warmup_iters > 0:
        logging.info(
            "[G3_warmup] density_net will be frozen for the first %d iters.",
            gating_warmup_iters,
        )
        # 시작 시점: density_net 동결
        for p in model.inr.density_net.parameters():
            p.requires_grad_(False)
    # ===== [G3_warmup 끝] =====

    # ===== [FF Loss 추가] FF Loss 활성화 여부 및 패치 샘플링 하이퍼파라미터 =====
    use_ff_loss = model.ff_loss_fn is not None
    patch_size = getattr(args, "patch_size", 16)
    n_patches  = getattr(args, "n_patches",  8)

    # ===== [Hard Slice Mining] 슬라이스별 MSE 잔차 EMA 초기화 =====
    SLICE_RESIDUAL_EMA = 0.99
    n_slices = dataset.n_slices
    slice_residuals = torch.zeros(n_slices, dtype=torch.float32)
    slice_counts    = torch.zeros(n_slices, dtype=torch.long)
    _residuals_initialized = False
    logging.info("[E3] slice_residuals will be initialized from first batch MSE (no warmup needed).")
    # ===== [Hard Slice Mining 끝] =====

    HARD_MINING_WARMUP = getattr(args, "hard_mining_warmup", 0)
    # ===== [HM2] --hard-mining-main-loss 플래그: get_batch()에도 hard mining 적용 여부 =====
    use_hard_mining_main_loss = getattr(args, "hard_mining_main_loss", False)
    if use_hard_mining_main_loss:
        logging.info(
            "[HM2] Hard Mining for main MSE loss (get_batch) ENABLED. "
            "Slice residuals will be used to weight pixel sampling after warmup."
        )
    else:
        logging.info(
            "[HM2] Hard Mining for main MSE loss (get_batch) DISABLED (default). "
            "Use --hard-mining-main-loss to enable."
        )
    # ===== [HM2 끝] =====

    logging.info(
        "[E3] Hard Slice Mining warmup: %d iters (0 = immediate activation after first-batch init).",
        HARD_MINING_WARMUP,
    )

    if use_ff_loss:
        logging.info(
            "FF Loss enabled: weight=%.4f, patch_size=%d, n_patches=%d",
            loss_weights[FF_LOSS], patch_size, n_patches,
        )

    average = MovingAverage(1 - 0.001)
    logging_header = False
    logging.info("NeSVoR training starts.")
    train_time = 0.0
    for i in range(1, args.n_iter + 1):
        train_step_start = time.time()

        # ===== [G3_warmup] warmup 종료 시점에 density_net 해동 =====
        if gating_warmup_iters > 0 and i == gating_warmup_iters + 1:
            for p in model.inr.density_net.parameters():
                p.requires_grad_(True)
            logging.info(
                "[G3_warmup] density_net unfrozen at iter %d.", i
            )
        # ===== [G3_warmup 끝] =====

        # ===== [HM2] get_batch() 호출 시 sampling_probs 결정 =====
        if use_hard_mining_main_loss and _residuals_initialized and i > HARD_MINING_WARMUP:
            main_sampling_probs = slice_residuals / slice_residuals.sum()
        else:
            main_sampling_probs = None
        # ===== [HM2 끝] =====

        # forward
        batch = dataset.get_batch(args.batch_size, args.device, sampling_probs=main_sampling_probs)
        with torch.cuda.amp.autocast(fp16):
            losses = model(**batch)

            # ===== [Hard Slice Mining] SLICE_MSE_KEY 관리 =====
            if SLICE_MSE_KEY in losses:
                pixel_mse_vals, s_idx = losses.pop(SLICE_MSE_KEY)

                if not _residuals_initialized:
                    for si in s_idx.unique():
                        mask = (s_idx == si)
                        slice_residuals[si.item()] = pixel_mse_vals[mask].mean().item()
                        slice_counts[si.item()] = 1
                    seen = slice_counts > 0
                    if seen.any():
                        init_mean = slice_residuals[seen].mean().item()
                        slice_residuals[~seen] = init_mean
                    _residuals_initialized = True
                    logging.info(
                        "[E3] slice_residuals initialized from first batch. "
                        "mean=%.6f, min=%.6f, max=%.6f",
                        slice_residuals.mean().item(),
                        slice_residuals.min().item(),
                        slice_residuals.max().item(),
                    )

                for si in s_idx.unique():
                    mask = (s_idx == si)
                    mean_mse = pixel_mse_vals[mask].mean().item()
                    slice_residuals[si.item()] = (
                        SLICE_RESIDUAL_EMA * slice_residuals[si.item()]
                        + (1 - SLICE_RESIDUAL_EMA) * mean_mse
                    )
            # ===== [Hard Slice Mining 끝] =====

            # ===== [FF Loss 추가] 패치 배치 샘플링 및 patch_forward 호출 =====
            if use_ff_loss:
                if i <= HARD_MINING_WARMUP or not _residuals_initialized:
                    sampling_probs = None
                else:
                    sampling_probs = slice_residuals / slice_residuals.sum()
                patch_batch = dataset.get_patch_batch(
                    n_patches, patch_size, args.device,
                    sampling_probs=sampling_probs,
                )
                if patch_batch:
                    patch_losses = model.patch_forward(**patch_batch)
                    losses.update(patch_losses)
            # ===== [FF Loss 추가 끝] =====

            # ===== [G3/B] Diversity Loss 계산 =====
            if use_diversity_loss:
                _target = torch.tensor(
                    target_diversity_var,
                    dtype=_level_weights_param.dtype,
                    device=_level_weights_param.device,
                )

                if diversity_loss_fn == "entropy":
                    # [B] softmax 훈 엔트로피 기준
                    # entropy 최대 = 균등 분포 (= 분화 없음) → 낮이는 방향이 의도
                    # relu(entropy - target): entropy > target 일 때만 패널티 부과
                    _p = F.softmax(_level_weights_param, dim=0)
                    _entropy = -(_p * (_p + 1e-8).log()).sum()
                    losses[DIVERSITY_LOSS] = torch.relu(_entropy - _target)

                elif diversity_loss_fn == "gini":
                    # [B] softmax 훈 Gini 비순수 기준
                    # gini 최대 = 균등 분포 (= 분화 없음) → 낮이는 방향이 의도
                    # relu(gini - target): gini > target 일 때만 패널티 부과
                    _p = F.softmax(_level_weights_param, dim=0)
                    _gini = 1.0 - (_p ** 2).sum()
                    losses[DIVERSITY_LOSS] = torch.relu(_gini - _target)

                else:
                    # [G3/G5_lowvar] variance 기준 (default, G5_lowvar 호환)
                    # variance 높을수록 분화 우수 → 높이는 방향이 의도
                    # relu(target - var): var < target 일 때만 패널티 부과
                    if diversity_loss_space == "softmax":
                        _w = F.softmax(_level_weights_param, dim=0) * model.inr.n_levels
                        _var = torch.var(_w)
                    else:
                        # raw logit 기준 (G3_softvar 기본)
                        _var = torch.var(_level_weights_param)
                    losses[DIVERSITY_LOSS] = torch.relu(_target - _var)
            # ===== [G3/B 끝] =====

            loss = 0
            for k in losses:
                if k in loss_weights and loss_weights[k]:
                    loss = loss + loss_weights[k] * losses[k]
        # backward
        scaler.scale(loss).backward()

        # ===== [G3] gating 그룹 gradient clipping =====
        if gating_grad_clip > 0.0 and params_gating:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params_gating, max_norm=gating_grad_clip)
        # ===== [G3 끝] =====

        if args.debug:
            for _name, _p in model.named_parameters():
                if _p.grad is not None and not _p.grad.isfinite().all():
                    logging.warning("iter %d: Found NaNs in the grad of %s", i, _name)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        train_time += time.time() - train_step_start
        for k in losses:
            average(k, losses[k].item())
        if (decay_milestones and i >= decay_milestones[0]) or i == args.n_iter:
            if not logging_header:
                train_logger = TrainLogger(
                    "time",
                    "epoch",
                    "iter",
                    *list(losses.keys()),
                    "lr",
                )
                logging_header = True
            train_logger.log(
                datetime.timedelta(seconds=int(train_time)),
                dataset.epoch,
                i,
                *[average[k] for k in losses],
                optimizer.param_groups[0]["lr"],
            )
            if i < args.n_iter:
                decay_milestones.pop(0)
                scheduler.step()
            if scaler.is_enabled():
                current_scaler = scaler.get_scale()
                if current_scaler < 1 / (2**5):
                    logging.warning(
                        "Numerical instability detected! "
                        "The scale of GradScaler is %f, which is too small. "
                        "The results might be suboptimal. "
                        "Try to set --single-precision or run the command again with a different random seed."
                    )
                if i == args.n_iter:
                    logging.debug("Final scale of GradScaler = %f" % current_scaler)

    # outputs
    transformation = model.transformation
    ax = transformation.axisangle()
    ax[:, -3:] *= spatial_scaling
    transformation = RigidTransform(ax)
    transformation = RigidTransform(
        torch.cat([torch.zeros_like(center), center])[None]
    ).compose(transformation)
    model.inr.bounding_box.copy_(bb)
    dataset.xyz *= spatial_scaling

    dataset.transformation = transformation
    mask = dataset.mask
    output_slices = []
    for i in range(len(slices)):
        output_slice = slices[i].clone()
        output_slice.transformation = transformation[i]
        output_slices.append(output_slice)
    return model.inr, output_slices, mask
