import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import data_loader
import noise_utils
import reading_utils
from losses import get_loss
from rollout_utils import RolloutEvaluator, validation
from simulator_factory import _get_simulator
from train_config import (
    INPUT_SEQUENCE_LENGTH,
    KINEMATIC_PARTICLE_ID,
    Config,
)
from train_paths import _resolve_model_path
from train_utils import (
    _AMP_AUTOCAST_SUPPORTS_DEVICE_TYPE,
    _AMP_GRADSCALER_SUPPORTS_DEVICE_TYPE,
    _cleanup_distributed,
    GradScaler,
    _compute_grad_norm,
    _format_eta,
    _launch_tensorboard,
    _resolve_amp_dtype,
    _resolve_rollout_dataset_path,
    _set_seed,
    autocast,
    optimizer_to,
    save_model_and_train_state,
)

if TYPE_CHECKING:  # type hints only; no runtime dependency when disabled
    from torch.utils.tensorboard import SummaryWriter


def train(cfg: Config, device: torch.device):
    distributed = cfg.world_size > 1
    is_main_process = cfg.rank == 0

    accum_steps = max(1, int(cfg.gradient_accumulation_steps))
    if accum_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")

    _set_seed(cfg.seed + cfg.rank)

    metadata_key = (
        cfg.active_scenario.metadata_split if cfg.active_scenario else "train"
    )
    metadata = reading_utils.read_metadata(cfg.data_path, metadata_key)

    simulator_core = _get_simulator(metadata, cfg.noise_std, cfg.noise_std, device, cfg)
    simulator_core.to(device)
    noise_sampler = noise_utils.get_noise(cfg.noise)
    loss_fn = get_loss(cfg.loss)

    resume_state: dict[str, Any] | None = None
    step = 0
    epoch = 0
    train_loss_hist: list[tuple[int, float]] = []
    valid_loss_hist: list[tuple[int, float]] = []
    latest_valid_loss_value: float | None = None
    latest_rollout_metrics: dict[str, float | None] | None = None
    valid_loss: torch.Tensor | None = None
    train_loss = 0.0

    if cfg.model_file is not None:
        model_path = _resolve_model_path(cfg)
        train_state_path = None
        if cfg.train_state_file == "latest":
            import re

            match = re.match(r".*model-(\d+)\.pt", model_path)
            if match:
                step_num = int(match.groups()[0])
                candidate = Path(cfg.model_path) / f"train_state-{step_num}.pt"
                if candidate.exists():
                    train_state_path = candidate
        elif cfg.train_state_file:
            candidate = Path(cfg.model_path) / cfg.train_state_file
            if candidate.exists():
                train_state_path = candidate

        if (
            Path(model_path).exists()
            and train_state_path
            and Path(train_state_path).exists()
        ):
            if is_main_process:
                print(f"Resume from: {model_path}, {train_state_path}")
            simulator_core.load(model_path)
            resume_state = torch.load(train_state_path, map_location=device)
            step = int(resume_state["global_train_state"]["step"])
            epoch = int(resume_state["global_train_state"]["epoch"])
            train_loss_hist = list(resume_state["loss_history"]["train"])
            valid_loss_hist = list(resume_state["loss_history"]["valid"])
        else:
            if is_main_process:
                print("Resume files not fully found; starting fresh.")

    if distributed:
        ddp_kwargs: dict[str, Any] = {
            "broadcast_buffers": False,
            "find_unused_parameters": cfg.ddp_find_unused_parameters,
        }
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [device.index]
            ddp_kwargs["output_device"] = device.index
        simulator: Any = DDP(simulator_core, **ddp_kwargs)
        # 互換性のため、DDP ラッパーにもメソッドを生やしておく
        simulator.predict_positions = simulator.module.predict_positions
        simulator.predict_accelerations = simulator.module.predict_accelerations
    else:
        simulator = simulator_core

    optimizer = torch.optim.Adam(simulator.parameters(), lr=cfg.lr_init)
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer_state"])
        optimizer_to(optimizer, device)
        current_lr = optimizer.param_groups[0].get("lr", cfg.lr_init)
    else:
        current_lr = cfg.lr_init
    optimizer.zero_grad(set_to_none=True)

    amp_enabled = bool(cfg.amp_enable and device.type == "cuda")
    amp_dtype = _resolve_amp_dtype(cfg.amp_dtype) if amp_enabled else torch.float32
    # device_type 引数付きの GradScaler は一部バージョンで初回 step で
    # "No inf checks were recorded" を誤検知することがあるため、従来 API に統一する。
    scaler = GradScaler(enabled=amp_enabled)

    log_interval = max(1, int(cfg.log_interval))
    tensorboard_interval = (
        max(1, int(cfg.tensorboard_interval))
        if cfg.tensorboard_interval is not None
        else log_interval
    )

    tb_writer: "SummaryWriter | None" = None  # type: ignore[name-defined]
    tb_server: Any | None = None
    tb_url: str | None = None
    if is_main_process and cfg.tensorboard_enable:
        # 遅延インポートで、無効化時に依存パッケージを要求しない
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
        tb_log_base = (
            Path(cfg.tensorboard_log_dir)
            if cfg.tensorboard_log_dir is not None
            else Path(cfg.model_path) / "tb"
        )
        try:
            tb_writer = SummaryWriter(log_dir=str(tb_log_base))
            tb_log_path = Path(tb_writer.log_dir).resolve()
            print(f"TensorBoard logging to {tb_log_path}")
            tb_server, tb_url = _launch_tensorboard(
                tb_log_path, cfg.tensorboard_host, cfg.tensorboard_port
            )
            if tb_url:
                print(f"TensorBoard available at {tb_url}")
            else:
                cmd_hint = f"tensorboard --logdir {tb_log_path}"
                print(f"Launch TensorBoard manually if needed:\n  {cmd_hint}")
        except Exception as exc:  # pragma: no cover
            print(f"Warning: could not initialize TensorBoard writer: {exc}")
            tb_writer = None
            tb_server = None

    train_dataset = data_loader.SamplesDataset(
        path=Path(cfg.data_path) / "train.npz",
        input_length_sequence=INPUT_SEQUENCE_LENGTH,
        fraction=cfg.train_dataset_fraction,
        max_trajectories=cfg.train_dataset_count,
    )
    feature_sample = train_dataset[0]
    features_sample = (
        feature_sample[0]
        if isinstance(feature_sample, (tuple, list))
        else feature_sample
    )
    if not isinstance(features_sample, (tuple, list)):
        msg = "Unexpected dataset sample structure; expected (features, label)."
        raise TypeError(msg)
    feature_components = len(features_sample)

    drop_last = accum_steps > 1
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=cfg.world_size,
            rank=cfg.rank,
            shuffle=True,
            drop_last=drop_last,
        )
        if distributed
        else None
    )
    train_loader_kwargs: dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "shuffle": train_sampler is None,
        "sampler": train_sampler,
        "num_workers": cfg.num_workers,
        "persistent_workers": cfg.persistent_workers if cfg.num_workers > 0 else False,
        "pin_memory": cfg.pin_memory,
        "drop_last": drop_last,
        "collate_fn": data_loader.collate_fn,
    }
    if cfg.prefetch_factor is not None and cfg.num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = cfg.prefetch_factor
    dl = torch.utils.data.DataLoader(train_dataset, **train_loader_kwargs)

    train_start_time = time.perf_counter()
    ema_step_time: float | None = None
    steps_counted_for_eta = 0

    validation_interval = cfg.validation_interval
    dl_valid = None
    dl_valid_iter = None
    if is_main_process and validation_interval is not None:
        valid_dataset = data_loader.SamplesDataset(
            path=Path(cfg.data_path) / "valid.npz",
            input_length_sequence=INPUT_SEQUENCE_LENGTH,
            fraction=cfg.valid_dataset_fraction,
            max_trajectories=cfg.valid_dataset_count,
        )
        if len(valid_dataset[0][0]) != feature_components:
            msg = "`valid.npz` と `train.npz` の特徴数が一致していません。"
            raise ValueError(msg)
        valid_loader_kwargs: dict[str, Any] = {
            "batch_size": cfg.batch_size,
            "shuffle": False,
            "num_workers": cfg.num_workers,
            "persistent_workers": cfg.persistent_workers
            if cfg.num_workers > 0
            else False,
            "pin_memory": cfg.pin_memory,
            "drop_last": False,
            "collate_fn": data_loader.collate_fn,
        }
        if cfg.prefetch_factor is not None and cfg.num_workers > 0:
            valid_loader_kwargs["prefetch_factor"] = cfg.prefetch_factor
        dl_valid = torch.utils.data.DataLoader(valid_dataset, **valid_loader_kwargs)
        dl_valid_iter = iter(dl_valid)

    def _next_valid_example():
        nonlocal dl_valid_iter
        if dl_valid is None or dl_valid_iter is None:
            return None
        try:
            return next(dl_valid_iter)
        except StopIteration:
            dl_valid_iter = iter(dl_valid)
            return next(dl_valid_iter)

    rollout_evaluator: RolloutEvaluator | None = None
    rollout_interval = (
        int(cfg.rollout_interval)
        if cfg.rollout_interval and cfg.rollout_interval > 0 and is_main_process
        else None
    )
    if rollout_interval is not None:
        rollout_dataset_path = _resolve_rollout_dataset_path(cfg)
        if rollout_dataset_path is None:
            if is_main_process:
                print(
                    "Warning: rollout metrics enabled but no suitable dataset was found; disabling rollout evaluation."
                )
            rollout_interval = None
        else:
            rollout_loader = data_loader.get_data_loader_by_trajectories(
                rollout_dataset_path
            )
            rollout_evaluator = RolloutEvaluator(
                rollout_loader, device, cfg.rollout_max_examples
            )
            if is_main_process:
                print(
                    f"Rollout metrics every {rollout_interval} steps using {rollout_dataset_path}"
                )

    epoch_train_loss = 0.0
    epoch_valid_loss: torch.Tensor | None = None
    steps_this_epoch = 0
    last_grad_norm: float | None = None
    micro_loss_accum = 0.0
    micro_batch_count = 0
    micro_accum_start: float | None = None

    def _optimizer_step() -> None:
        nonlocal step
        nonlocal current_lr
        nonlocal last_grad_norm
        nonlocal steps_this_epoch
        nonlocal epoch_train_loss
        nonlocal micro_loss_accum
        nonlocal micro_batch_count
        nonlocal train_loss
        nonlocal ema_step_time
        nonlocal steps_counted_for_eta
        nonlocal micro_accum_start
        nonlocal valid_loss
        nonlocal latest_valid_loss_value
        nonlocal latest_rollout_metrics

        if step >= cfg.ntraining_steps:
            return

        scaler.unscale_(optimizer)
        parameters = (
            simulator.module.parameters()
            if isinstance(simulator, DDP)
            else simulator.parameters()
        )
        last_grad_norm = _compute_grad_norm(parameters)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        mean_loss = micro_loss_accum / max(1, micro_batch_count)
        train_loss = float(mean_loss)
        epoch_train_loss += train_loss
        micro_loss_accum = 0.0
        micro_batch_count = 0
        steps_this_epoch += 1

        lr_new = cfg.lr_init * (cfg.lr_decay ** (step / cfg.lr_decay_steps))
        for group in optimizer.param_groups:
            group["lr"] = lr_new
        current_lr = lr_new

        step += 1

        if micro_accum_start is not None:
            step_duration = time.perf_counter() - micro_accum_start
            micro_accum_start = None
            if ema_step_time is None:
                ema_step_time = step_duration
            else:
                alpha = 0.05
                ema_step_time = alpha * step_duration + (1 - alpha) * ema_step_time
            steps_counted_for_eta += 1

        if (
            is_main_process
            and validation_interval is not None
            and validation_interval > 0
            and step > 0
            and step % validation_interval == 0
        ):
            sampled_valid_example = _next_valid_example()
            if sampled_valid_example is not None:
                with torch.no_grad():
                    valid_loss_tensor = validation(
                        simulator,
                        sampled_valid_example,
                        feature_components,
                        cfg,
                        device,
                    )
                valid_loss = valid_loss_tensor
                latest_valid_loss_value = float(valid_loss_tensor.item())
                print(f"[valid @ step {step}] {latest_valid_loss_value:.6f}")
                if tb_writer is not None:
                    tb_writer.add_scalar("valid/loss", latest_valid_loss_value, step)
            else:
                print("Warning: validation loader is empty; skipping validation step.")

        if (
            rollout_evaluator is not None
            and rollout_interval is not None
            and step > 0
            and step % rollout_interval == 0
        ):
            simulator.eval()
            with torch.no_grad():
                rollout_metrics = rollout_evaluator.evaluate(simulator)
            simulator.train()
            latest_rollout_metrics = rollout_metrics
            if (
                rollout_metrics["rollout_rmse_mean"] is not None
                and rollout_metrics["rollout_rmse_last"] is not None
            ):
                print(
                    "[rollout @ step "
                    f"{step}] mean={rollout_metrics['rollout_rmse_mean']:.6f} "
                    f"last={rollout_metrics['rollout_rmse_last']:.6f} "
                    f"instability={rollout_metrics['rollout_instability']:.3f}"
                )
                if tb_writer is not None:
                    tb_writer.add_scalar(
                        "rollout/rmse_mean",
                        float(rollout_metrics["rollout_rmse_mean"]),
                        step,
                    )
                    tb_writer.add_scalar(
                        "rollout/rmse_last",
                        float(rollout_metrics["rollout_rmse_last"]),
                        step,
                    )
                    tb_writer.add_scalar(
                        "rollout/instability",
                        float(rollout_metrics["rollout_instability"]),
                        step,
                    )

        if is_main_process and step % log_interval == 0:
            remaining_steps = max(0, cfg.ntraining_steps - step)
            if ema_step_time is not None and steps_counted_for_eta > 0:
                eta_sec = remaining_steps * ema_step_time
            else:
                elapsed = time.perf_counter() - train_start_time
                avg = elapsed / max(1, steps_counted_for_eta)
                eta_sec = remaining_steps * avg
            eta_str = _format_eta(eta_sec)
            log_parts = [
                f"epoch={epoch}",
                f"step={step}/{cfg.ntraining_steps}",
                f"loss={train_loss:.6f}",
                f"lr={current_lr:.6e}",
            ]
            if last_grad_norm is not None:
                log_parts.append(f"grad_norm={last_grad_norm:.4f}")
            if latest_valid_loss_value is not None:
                log_parts.append(f"valid={latest_valid_loss_value:.6f}")
            if latest_rollout_metrics is not None:
                rmse_mean = latest_rollout_metrics.get("rollout_rmse_mean")
                rmse_last = latest_rollout_metrics.get("rollout_rmse_last")
                instability = latest_rollout_metrics.get("rollout_instability")
                if rmse_mean is not None and rmse_last is not None:
                    log_parts.append(f"rollout_mean={rmse_mean:.6f}")
                    log_parts.append(f"rollout_last={rmse_last:.6f}")
                    log_parts.append(f"rollout_instab={instability:.3f}")
            log_parts.append(f"eta={eta_str}")
            print(" ".join(log_parts))

        if (
            is_main_process
            and tb_writer is not None
            and step % tensorboard_interval == 0
        ):
            tb_writer.add_scalar("train/loss", train_loss, step)
            tb_writer.add_scalar("train/lr", current_lr, step)
            if last_grad_norm is not None:
                tb_writer.add_scalar("train/grad_norm", last_grad_norm, step)
            if latest_valid_loss_value is not None:
                tb_writer.add_scalar("valid/loss", latest_valid_loss_value, step)
            if (
                latest_rollout_metrics is not None
                and latest_rollout_metrics.get("rollout_rmse_mean") is not None
            ):
                tb_writer.add_scalar(
                    "rollout/rmse_mean",
                    float(latest_rollout_metrics["rollout_rmse_mean"]),
                    step,
                )
                tb_writer.add_scalar(
                    "rollout/rmse_last",
                    float(latest_rollout_metrics["rollout_rmse_last"]),
                    step,
                )
                tb_writer.add_scalar(
                    "rollout/instability",
                    float(latest_rollout_metrics["rollout_instability"]),
                    step,
                )
            tb_writer.flush()

        if is_main_process and cfg.nsave_steps and step % cfg.nsave_steps == 0:
            save_model_and_train_state(
                simulator,
                cfg,
                step,
                epoch,
                optimizer,
                train_loss,
                float(valid_loss.item()) if valid_loss is not None else None,
                train_loss_hist,
                valid_loss_hist,
            )

    try:
        while step < cfg.ntraining_steps:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for example in dl:
                if step >= cfg.ntraining_steps:
                    break
                if micro_batch_count == 0:
                    micro_accum_start = time.perf_counter()

                features, labels = example
                position = features[0].to(device)
                particle_type = features[1].to(device)
                if len(features) == 4:
                    material_property = features[2].to(device)
                    n_particles_per_example = features[3].to(device)
                elif len(features) == 3:
                    material_property = None
                    n_particles_per_example = features[2].to(device)
                else:
                    raise NotImplementedError
                labels = labels.to(device)

                sampled_noise = noise_sampler(
                    position, noise_std_last_step=cfg.noise_std
                ).to(device)
                non_kinematic_mask = (particle_type != KINEMATIC_PARTICLE_ID).to(device)
                sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

                autocast_kwargs: dict[str, Any] = {
                    "enabled": amp_enabled,
                }
                if amp_enabled:
                    autocast_kwargs["dtype"] = amp_dtype
                if _AMP_AUTOCAST_SUPPORTS_DEVICE_TYPE:
                    autocast_kwargs["device_type"] = device.type
                autocast_ctx = autocast(**autocast_kwargs)
                with autocast_ctx:
                    pred_acc, target_acc = simulator(
                        next_positions=labels,
                        position_sequence_noise=sampled_noise,
                        position_sequence=position,
                        nparticles_per_example=n_particles_per_example,
                        particle_types=particle_type,
                        material_property=material_property,
                    )
                    loss_tensor = loss_fn(pred_acc, target_acc, non_kinematic_mask)

                train_loss_micro = float(loss_tensor.item())
                micro_loss_accum += train_loss_micro
                micro_batch_count += 1

                scaled_loss = loss_tensor / accum_steps
                scaler.scale(scaled_loss).backward()

                if micro_batch_count >= accum_steps:
                    _optimizer_step()

            if step >= cfg.ntraining_steps:
                break

            if micro_batch_count > 0 and step < cfg.ntraining_steps:
                _optimizer_step()

            epoch_avg = (
                (epoch_train_loss / steps_this_epoch) if steps_this_epoch > 0 else 0.0
            )
            if is_main_process:
                train_loss_hist.append((epoch, float(epoch_avg)))
                if dl_valid is not None and validation_interval is not None:
                    sampled_valid_example = _next_valid_example()
                    if sampled_valid_example is not None:
                        with torch.no_grad():
                            epoch_valid_loss = validation(
                                simulator,
                                sampled_valid_example,
                                feature_components,
                                cfg,
                                device,
                            )
                        epoch_valid_loss_float = float(epoch_valid_loss.item())
                        valid_loss_hist.append((epoch, epoch_valid_loss_float))
                        latest_valid_loss_value = epoch_valid_loss_float
                        print(
                            f"[epoch {epoch}] train={epoch_avg:.6f} valid={epoch_valid_loss_float:.6f}"
                        )
                        if tb_writer is not None:
                            tb_writer.add_scalar(
                                "epoch/valid_loss", epoch_valid_loss_float, epoch
                            )
                    else:
                        print(
                            f"[epoch {epoch}] train={epoch_avg:.6f} (validation skipped: dataset empty)"
                        )
                else:
                    print(f"[epoch {epoch}] train={epoch_avg:.6f}")
                if tb_writer is not None:
                    tb_writer.add_scalar("epoch/train_loss", epoch_avg, epoch)
                    tb_writer.flush()

            epoch_train_loss = 0.0
            steps_this_epoch = 0
            epoch += 1

            if step >= cfg.ntraining_steps:
                break

    except KeyboardInterrupt:
        if is_main_process:
            print("Interrupted. Saving last state...")
    finally:
        if distributed:
            dist.barrier()
        if is_main_process:
            save_model_and_train_state(
                simulator,
                cfg,
                step,
                epoch,
                optimizer,
                train_loss,
                float(valid_loss.item()) if valid_loss is not None else None,
                train_loss_hist,
                valid_loss_hist,
            )
            if tb_writer is not None:
                tb_writer.close()
            if (
                tb_server is not None
                and hasattr(tb_server, "_server")
                and tb_server._server is not None
            ):
                try:  # pragma: no cover
                    tb_server._server.stop()
                except Exception:
                    pass
        if distributed:
            _cleanup_distributed()


__all__ = ["train"]
