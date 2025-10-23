import glob
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time

import torch
import yaml
from tqdm import tqdm

import data_loader

# --- gns モジュール群(あなたの環境のパスに応じて調整してください) ---
import learned_simulator
import noise_utils
import reading_utils

# -----------------------
# 設定
# -----------------------
INPUT_SEQUENCE_LENGTH = 6  # 先頭6ステップを履歴として使う(速度5本が引ける)
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3


@dataclass
class Config:
    mode: str = "train"  # train / valid / rollout
    data_path: str = "./data/"
    model_path: str = "./models/"
    output_path: str = "./rollouts/"
    output_filename: str = "rollout"

    batch_size: int = 2
    noise_std: float = 6.7e-4
    log_interval: int = 1000

    ntraining_steps: int = int(2e7)
    validation_interval: int | None = None
    nsave_steps: int = 5000

    lr_init: float = 1e-4
    lr_decay: float = 0.1
    lr_decay_steps: int = int(5e6)

    model_file: str | None = None  # "latest" or ファイル名 or null
    train_state_file: str | None = "train_state.pt"  # "latest" or ファイル名 or null

    cuda_device_number: int | None = None  # null で自動選択

    # ---------- 追加: 研究ループ/高速化オプション ----------
    # モデル構成の上書き（未指定ならデフォルト）
    nmessage_passing_steps: int | None = None
    latent_dim: int | None = None
    mlp_hidden_dim: int | None = None
    particle_type_embedding_size: int | None = None
    connectivity_radius: float | None = None

    # DataLoader/I-O
    num_workers: int = 0
    persistent_workers: bool = False
    pin_memory: bool = True
    prefetch_factor: int | None = None
    train_scenes: int | None = None  # 例: 64, None で制限なし
    valid_scenes: int | None = None

    # 学習制御
    grad_accum: int = 1
    amp: str | None = None  # "fp16" / "bf16" / None
    seed: int | None = None

    # 評価と早期終了
    eval_rollout_steps: int = 0  # 0 で one-step のみ
    early_stop_check_step: int | None = None
    early_stop_required_improvement: float = 0.3  # 30% 改善を要求

    # チェックポイント管理
    ckpt_keep: int | None = None  # None で無制限


def load_config(path: str) -> Config:
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


# -----------------------
# 共通ユーティリティ
# -----------------------
def optimizer_to(optim: torch.optim.Optimizer, device: torch.device):
    """オプティマイザ内部のテンソルを所定デバイスへ移動"""
    for state in optim.state.values():
        if isinstance(state, torch.Tensor):
            state.data = state.data.to(device)
            if state._grad is not None:
                state._grad.data = state._grad.data.to(device)
        elif isinstance(state, dict):
            for sub in state.values():
                if isinstance(sub, torch.Tensor):
                    sub.data = sub.data.to(device)
                    if sub._grad is not None:
                        sub._grad.data = sub._grad.data.to(device)


def acceleration_loss(
    pred_acc: torch.Tensor,
    target_acc: torch.Tensor,
    non_kinematic_mask: torch.Tensor,  # 学習に用いる粒子がtrue
) -> torch.Tensor:  # 加速度のロス計算
    loss = (pred_acc - target_acc) ** 2
    loss = loss.sum(dim=-1)  # D 次元を和
    num_non_kinematic = non_kinematic_mask.sum()  # 学習に用いる粒子数
    masked = torch.where(
        non_kinematic_mask.bool(), loss, torch.zeros_like(loss)
    )  # 学習に用いる粒子のみロスを考慮
    return masked.sum() / num_non_kinematic.clamp(
        min=1
    )  # 平均を取る(clampはゼロ除算防止)


def save_model_and_train_state(
    simulator: learned_simulator.LearnedSimulator,
    cfg: Config,
    step: int,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    train_loss: float | None,
    valid_loss: float | None,
    train_loss_hist: list[tuple[int, float]],
    valid_loss_hist: list[tuple[int, float]],
):
    Path(cfg.model_path).mkdir(parents=True, exist_ok=True)
    # モデル保存
    model_fname = Path(cfg.model_path) / f"model-{step}.pt"
    simulator.save(str(model_fname))

    # train_state 保存
    train_state = dict(
        optimizer_state=optimizer.state_dict(),
        global_train_state={
            "step": step,
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        },
        loss_history={"train": train_loss_hist, "valid": valid_loss_hist},
        used_config=_asjsonable_cfg(cfg),
    )
    ts_fname = Path(cfg.model_path) / f"train_state-{step}.pt"
    torch.save(train_state, ts_fname)


def _asjsonable_cfg(cfg: Config) -> dict[str, Any]:
    # dataclass → dict(JSON化しやすい形に変換)
    return {
        "mode": cfg.mode,
        "data_path": cfg.data_path,
        "model_path": cfg.model_path,
        "output_path": cfg.output_path,
        "output_filename": cfg.output_filename,
        "batch_size": cfg.batch_size,
        "noise_std": cfg.noise_std,
        "log_interval": cfg.log_interval,
        "ntraining_steps": cfg.ntraining_steps,
        "validation_interval": cfg.validation_interval,
        "nsave_steps": cfg.nsave_steps,
        "lr_init": cfg.lr_init,
        "lr_decay": cfg.lr_decay,
        "lr_decay_steps": cfg.lr_decay_steps,
        "model_file": cfg.model_file,
        "train_state_file": cfg.train_state_file,
        "cuda_device_number": cfg.cuda_device_number,
    }


def _format_eta(seconds: float) -> str:
    """秒を人間が読みやすい ETA 文字列に整形 (H:MM:SS or MM:SS)。"""
    sec = int(max(0, seconds))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# -----------------------
# モデル生成
# -----------------------
def _get_simulator(
    metadata: dict[str, Any],
    acc_noise_std: float,
    vel_noise_std: float,
    device: torch.device,
) -> learned_simulator.LearnedSimulator:
    """LearnedSimulator をメタ情報から生成"""
    normalization_stats = {
        "acceleration": {
            "mean": torch.FloatTensor(metadata["acc_mean"]).to(device),
            "std": torch.sqrt(
                torch.FloatTensor(metadata["acc_std"]) ** 2 + acc_noise_std**2
            ).to(device),
        },
        "velocity": {
            "mean": torch.FloatTensor(metadata["vel_mean"]).to(device),
            "std": torch.sqrt(
                torch.FloatTensor(metadata["vel_std"]) ** 2 + vel_noise_std**2
            ).to(device),
        },
    }

    if "nnode_in" in metadata and "nedge_in" in metadata:
        nnode_in = metadata["nnode_in"]
        nedge_in = metadata["nedge_in"]
    else:
        # 追加特徴がない標準構成のデフォルト（実データに合わせて必要なら要調整）
        nnode_in = 37 if metadata["dim"] == 3 else 30
        nedge_in = metadata["dim"] + 1

    # モデル上書きのためにグローバル cfg に触れる（main → train で生成されたもの）
    global CFG_FOR_BUILD
    latent_dim = CFG_FOR_BUILD.latent_dim if CFG_FOR_BUILD.latent_dim else 128
    nmessage = (
        CFG_FOR_BUILD.nmessage_passing_steps
        if CFG_FOR_BUILD.nmessage_passing_steps
        else 10
    )
    nmlp_layers = 2
    mlp_hidden_dim = (
        CFG_FOR_BUILD.mlp_hidden_dim if CFG_FOR_BUILD.mlp_hidden_dim else 128
    )
    particle_type_embedding_size = (
        CFG_FOR_BUILD.particle_type_embedding_size
        if CFG_FOR_BUILD.particle_type_embedding_size
        else 16
    )
    connectivity_radius = (
        CFG_FOR_BUILD.connectivity_radius
        if CFG_FOR_BUILD.connectivity_radius is not None
        else metadata["default_connectivity_radius"]
    )

    simulator = learned_simulator.LearnedSimulator(
        particle_dimensions=metadata["dim"],
        nnode_in=nnode_in,
        nedge_in=nedge_in,
        latent_dim=latent_dim,
        nmessage_passing_steps=nmessage,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
        connectivity_radius=connectivity_radius,
        # boundaries=np.array(metadata["bounds"]),
        normalization_stats=normalization_stats,
        nparticle_types=NUM_PARTICLE_TYPES,
        particle_type_embedding_size=particle_type_embedding_size,
        # boundary_clamp_limit=metadata.get("boundary_augment", 1.0),
        device=device,
    )
    return simulator


# -----------------------
# 予測系(rollout / validation / predict)
# -----------------------
def rollout(
    simulator: learned_simulator.LearnedSimulator,
    position: torch.Tensor,
    particle_types: torch.Tensor,
    material_property: torch.Tensor | None,
    n_particles_per_example: torch.Tensor,
    nsteps: int,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    """逐次1ステップ予測をシフト窓で積み上げるロールアウト
    position: (B, T, N, D) を想定。ここでは B=1 を前提に扱っている実装。
    """
    # B 次元を想定しつつコードは元実装を踏襲(B=1前提)
    initial_positions = position[:, :INPUT_SEQUENCE_LENGTH]
    ground_truth_positions = position[:, INPUT_SEQUENCE_LENGTH:]

    current_positions = initial_positions
    predictions_list = []

    for _ in tqdm(range(nsteps), total=nsteps):
        next_position = simulator.predict_positions(
            current_positions,
            nparticles_per_example=n_particles_per_example,
            particle_types=particle_types,
            material_property=material_property,
        )
        # 運動学粒子は教師で上書き
        kinematic_mask = (particle_types == KINEMATIC_PARTICLE_ID).to(device)
        next_position_gt = ground_truth_positions[:, _]  # (B=1, N, D)
        kinematic_mask = kinematic_mask.bool()[:, None].expand(
            -1, current_positions.shape[-1]
        )
        next_position = torch.where(kinematic_mask, next_position_gt, next_position)
        predictions_list.append(next_position)
        # シフト
        current_positions = torch.cat(
            [current_positions[:, 1:], next_position[:, None, :]], dim=1
        )

    predictions = torch.stack(predictions_list)  # (T', B, N, D) だが元実装互換に注意
    ground_truth_positions = ground_truth_positions.permute(1, 0, 2)

    loss = (predictions - ground_truth_positions) ** 2

    output = {
        "initial_positions": initial_positions.permute(1, 0, 2).cpu().numpy(),
        "predicted_rollout": predictions.cpu().numpy(),
        "ground_truth_rollout": ground_truth_positions.cpu().numpy(),
        "particle_types": particle_types.cpu().numpy(),
        "material_property": material_property.cpu().numpy()
        if material_property is not None
        else None,
    }
    return output, loss


def validation(
    simulator: learned_simulator.LearnedSimulator,
    example: tuple,
    n_features: int,
    cfg: Config,
    device: torch.device,
) -> torch.Tensor:
    position = example[0][0].to(device)
    particle_type = example[0][1].to(device)
    if n_features == 3:
        material_property = example[0][2].to(device)
        n_particles_per_example = example[0][3].to(device)
    elif n_features == 2:
        material_property = None
        n_particles_per_example = example[0][2].to(device)
    else:
        raise NotImplementedError
    labels = example[1].to(device)

    sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
        position, noise_std_last_step=cfg.noise_std
    ).to(device)
    non_kinematic_mask = (particle_type != KINEMATIC_PARTICLE_ID).to(device)
    sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

    with torch.no_grad():
        pred_acc, target_acc = simulator.predict_accelerations(
            next_positions=labels,
            position_sequence_noise=sampled_noise,
            position_sequence=position,
            nparticles_per_example=n_particles_per_example,
            particle_types=particle_type,
            material_property=material_property,
        )
    loss = acceleration_loss(pred_acc, target_acc, non_kinematic_mask)
    return loss


def predict(cfg: Config, device: torch.device):
    """valid/rollout モードの入口"""
    metadata = reading_utils.read_metadata(cfg.data_path, "rollout")
    simulator = _get_simulator(metadata, cfg.noise_std, cfg.noise_std, device)

    # モデル読み込み
    model_path = _resolve_model_path(cfg)
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model does not exist at {model_path}")
    simulator.load(model_path)
    simulator.to(device)
    simulator.eval()

    # 出力ディレクトリ
    Path(cfg.output_path).mkdir(parents=True, exist_ok=True)

    # valid.npz がなければ test.npz
    valid_npz = Path(cfg.data_path) / "valid.npz"
    split = "test" if (cfg.mode == "rollout" or (not valid_npz.exists())) else "valid"

    ds = data_loader.get_data_loader_by_trajectories(
        path=Path(cfg.data_path) / f"{split}.npz",
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor,
        limit_scenes=(cfg.valid_scenes if cfg.valid_scenes and cfg.valid_scenes > 0 else None),
    )

    # 特徴数(material_property の有無を判定)
    first = ds.dataset[0]
    if len(first) == 4:
        material_property_as_feature = True
    elif len(first) == 3:
        material_property_as_feature = False
    else:
        raise NotImplementedError

    eval_loss = []
    with torch.no_grad():
        for example_i, features in enumerate(ds):
            print(f"processing example number {example_i}")
            positions = features[0].to(device)
            # nsteps 計算
            if metadata.get("sequence_length") is not None:
                nsteps = int(metadata["sequence_length"]) - INPUT_SEQUENCE_LENGTH
            else:
                sequence_length = positions.shape[1]
                nsteps = int(sequence_length) - INPUT_SEQUENCE_LENGTH
            if cfg.eval_rollout_steps and cfg.eval_rollout_steps > 0:
                nsteps = max(0, min(nsteps, int(cfg.eval_rollout_steps)))

            particle_type = features[1].to(device)
            if material_property_as_feature:
                material_property = features[2].to(device)
                n_particles_per_example = torch.tensor(
                    [int(features[3])], dtype=torch.int32
                ).to(device)
            else:
                material_property = None
                n_particles_per_example = torch.tensor(
                    [int(features[2])], dtype=torch.int32
                ).to(device)

            example_rollout, loss = rollout(
                simulator,
                positions,
                particle_type,
                material_property,
                n_particles_per_example,
                nsteps,
                device,
            )
            example_rollout["metadata"] = metadata
            print(f"Predicting example {example_i} loss: {loss.mean()}")
            eval_loss.append(torch.flatten(loss))

            # ロールアウトの保存
            if cfg.mode == "rollout":
                example_rollout["loss"] = loss.mean()
                filename = f"{cfg.output_filename}_ex{example_i}.pkl"
                filename = str(Path(cfg.output_path) / filename)
                with Path(filename).open("wb") as f:
                    pickle.dump(example_rollout, f)

    print(f"Mean loss on rollout prediction: {torch.mean(torch.cat(eval_loss))}")


def _resolve_model_path(cfg: Config) -> str:
    """cfg.model_file が 'latest' のとき最新を解決。そうでなければ結合して返す。"""
    Path(cfg.model_path).mkdir(parents=True, exist_ok=True)
    if cfg.model_file == "latest":
        fnames = glob.glob(str(Path(cfg.model_path) / "model-*.pt"))
        if not fnames:
            msg = f"No model files found in {cfg.model_path}"
            raise FileNotFoundError(msg)
        expr = re.compile(r".*model-(\d+)\.pt")
        latest_num = -1
        latest_file = None
        for fname in fnames:
            m = expr.match(fname)
            if m:
                num = int(m.groups()[0])
                if num > latest_num:
                    latest_num = num
                    latest_file = fname
        if latest_file is None:
            msg = f"Could not resolve latest model in {cfg.model_path}"
            raise FileNotFoundError(msg)
        return latest_file
    if cfg.model_file:
        return str(Path(cfg.model_path) / cfg.model_file)
    # 明示されていない場合新規学習用なので存在しないパスを返す
    return str(Path(cfg.model_path) / "model-init.pt")


# -----------------------
# 学習
# -----------------------
def train(cfg: Config, device: torch.device):
    # メタ情報
    metadata = reading_utils.read_metadata(cfg.data_path, "train")
    train_loss = 0.0  # 初期化

    # モデルと最適化器
    simulator = _get_simulator(metadata, cfg.noise_std, cfg.noise_std, device)
    simulator.to(device)
    optimizer = torch.optim.Adam(simulator.parameters(), lr=cfg.lr_init)

    # AMP 設定
    amp_dtype: torch.dtype | None
    if cfg.amp == "fp16" and device.type == "cuda":
        amp_dtype = torch.float16
    elif cfg.amp == "bf16" and torch.cuda.is_available():
        # bf16 は GPU に依存。利用可能な場合にのみ使用。
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = None
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    log_interval = max(1, int(cfg.log_interval))

    # 進捗状態
    step = 0
    epoch = 0
    steps_per_epoch = 0

    valid_loss = None
    epoch_train_loss = 0.0
    epoch_valid_loss = None

    train_loss_hist = []
    valid_loss_hist = []

    # 再開処理
    if cfg.model_file is not None:
        model_path = _resolve_model_path(cfg)
        train_state_path = None
        if cfg.train_state_file == "latest":
            # モデルに合わせて同stepの train_state を探す
            m = re.match(r".*model-(\d+)\.pt", model_path)
            if m:
                step_num = int(m.groups()[0])
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
            print(f"Resume from: {model_path}, {train_state_path}")
            simulator.load(model_path)

            # Optimizer を作り直して state を読む（学習率などは後で上書き）
            optimizer = torch.optim.Adam(simulator.parameters())
            train_state = torch.load(train_state_path, map_location=device)
            optimizer.load_state_dict(train_state["optimizer_state"])
            optimizer_to(optimizer, device)

            step = int(train_state["global_train_state"]["step"])
            epoch = int(train_state["global_train_state"]["epoch"])
            train_loss_hist = list(train_state["loss_history"]["train"])
            valid_loss_hist = list(train_state["loss_history"]["valid"])
        else:
            print("Resume files not fully found; starting fresh.")

    simulator.train()

    # データローダ
    dl = data_loader.get_data_loader_by_samples(
        path=Path(cfg.data_path) / "train.npz",
        input_length_sequence=INPUT_SEQUENCE_LENGTH,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor,
        limit_scenes=(cfg.train_scenes if cfg.train_scenes and cfg.train_scenes > 0 else None),
    )
    n_features = len(dl.dataset[0])

    # ETA 用タイミング管理（軽量）
    train_start_time = time.perf_counter()
    ema_step_time: float | None = None
    steps_counted_for_eta = 0

    validation_interval = cfg.validation_interval
    dl_valid = None
    if validation_interval is not None:
        dl_valid = data_loader.get_data_loader_by_samples(
            path=Path(cfg.data_path) / "valid.npz",
            input_length_sequence=INPUT_SEQUENCE_LENGTH,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers,
            prefetch_factor=cfg.prefetch_factor,
            limit_scenes=(
                cfg.valid_scenes if cfg.valid_scenes and cfg.valid_scenes > 0 else None
            ),
        )
        if len(dl_valid.dataset[0]) != n_features:
            msg = "`valid.npz` と `train.npz` の特徴数が一致していません。"
            raise ValueError(msg)

    # 早期終了の参考用
    baseline_valid: float | None = None
    best_valid: float | None = None
    passed: bool | None = None

    def _maybe_prune_checkpoints():
        if cfg.ckpt_keep is None or cfg.ckpt_keep <= 0:
            return
        files = sorted(
            Path(cfg.model_path).glob("model-*.pt"),
            key=lambda p: int(re.findall(r"model-(\d+)\.pt", p.name)[0]) if re.findall(r"model-(\d+)\.pt", p.name) else -1,
        )
        if len(files) > cfg.ckpt_keep:
            for f in files[: len(files) - cfg.ckpt_keep]:
                try:
                    f.unlink()
                    tsf = f.with_name(f"train_state-{re.findall(r'model-(\d+)\\.pt', f.name)[0]}.pt")
                    if tsf.exists():
                        tsf.unlink()
                except Exception:
                    pass

    try:
        while step < cfg.ntraining_steps:
            for example in dl:
                step_tic = time.perf_counter()
                steps_per_epoch += 1
                # ((position, particle_type, [material_property], n_particles_per_example), labels)
                position = example[0][0].to(device)
                particle_type = example[0][1].to(device)
                if n_features == 3:
                    material_property = example[0][2].to(device)
                    n_particles_per_example = example[0][3].to(device)
                elif n_features == 2:
                    material_property = None
                    n_particles_per_example = example[0][2].to(device)
                else:
                    raise NotImplementedError
                labels = example[1].to(device)

                # ランダムウォーク雑音(非運動学粒子のみ)
                sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
                    position, noise_std_last_step=cfg.noise_std
                ).to(device)
                non_kinematic_mask = (particle_type != KINEMATIC_PARTICLE_ID).to(device)
                sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

                # 予測 & 教師(加速度)
                with torch.cuda.amp.autocast(enabled=(amp_dtype is not None), dtype=amp_dtype):
                    pred_acc, target_acc = simulator.predict_accelerations(
                        next_positions=labels,
                        position_sequence_noise=sampled_noise,
                        position_sequence=position,
                        nparticles_per_example=n_particles_per_example,
                        particle_types=particle_type,
                        material_property=material_property,
                    )

                # (オプション)軽量バリデーション
                if (
                    dl_valid is not None
                    and validation_interval is not None
                    and step > 0
                    and step % validation_interval == 0
                ):
                    sampled_valid_example = next(iter(dl_valid))
                    valid_loss = validation(
                        simulator, sampled_valid_example, n_features, cfg, device
                    )
                    print(f"[valid @ step {step}] {valid_loss.item():.6f}")

                # ロス → 逆伝播 → 更新
                loss = acceleration_loss(pred_acc, target_acc, non_kinematic_mask)
                train_loss = float(loss.item())
                epoch_train_loss += train_loss

                # 勾配蓄積
                accumulate = max(1, int(cfg.grad_accum))
                loss_to_backprop = loss / accumulate
                optimizer.zero_grad(set_to_none=True) if (step % accumulate == 0) else None
                if scaler.is_enabled():
                    scaler.scale(loss_to_backprop).backward()
                else:
                    loss_to_backprop.backward()

                # 勾配ノルム監視（簡易）
                grad_norm = None
                total_norm = 0.0
                for p in simulator.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2).item()
                        total_norm += param_norm * param_norm
                grad_norm = (total_norm ** 0.5) if total_norm > 0 else 0.0

                if (step + 1) % accumulate == 0:
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                # 学習率を指数減衰で更新
                lr_new = cfg.lr_init * (cfg.lr_decay ** (step / cfg.lr_decay_steps))
                for g in optimizer.param_groups:
                    g["lr"] = lr_new

                # 1 ステップ時間を EMA で更新（軽量）
                step_duration = time.perf_counter() - step_tic
                if ema_step_time is None:
                    ema_step_time = step_duration
                else:
                    alpha = 0.05  # 平滑化係数（小さいほど滑らか・低コスト）
                    ema_step_time = alpha * step_duration + (1 - alpha) * ema_step_time
                steps_counted_for_eta += 1

                if step % log_interval == 0:
                    remaining_steps = max(0, cfg.ntraining_steps - max(0, step))
                    if ema_step_time is not None and steps_counted_for_eta > 0:
                        eta_sec = remaining_steps * ema_step_time
                    else:
                        elapsed = time.perf_counter() - train_start_time
                        avg = elapsed / max(1, steps_counted_for_eta)
                        eta_sec = remaining_steps * avg
                    eta_str = _format_eta(eta_sec)
                    sps = (1.0 / ema_step_time) if (ema_step_time and ema_step_time > 0) else 0.0
                    print(
                        f"epoch={epoch} step={step}/{cfg.ntraining_steps} loss={train_loss:.6f} lr={lr_new:.6e} grad_norm={grad_norm:.2f} it/s={sps:.2f} eta={eta_str}"
                    )

                # 保存
                if cfg.nsave_steps and step % cfg.nsave_steps == 0:
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
                    _maybe_prune_checkpoints()

                # 早期停止の簡易ロジック（one-step valid ベース）
                if (
                    dl_valid is not None
                    and validation_interval is not None
                    and step > 0
                    and step % validation_interval == 0
                ):
                    sampled_valid_example = next(iter(dl_valid))
                    cur_valid = validation(
                        simulator, sampled_valid_example, n_features, cfg, device
                    )
                    v = float(cur_valid.item())
                    if baseline_valid is None:
                        baseline_valid = v
                        best_valid = v
                    else:
                        best_valid = min(best_valid, v) if best_valid is not None else v
                    # 指定ステップ到達かつ改善不足なら停止
                    if (
                        cfg.early_stop_check_step is not None
                        and step >= int(cfg.early_stop_check_step)
                        and baseline_valid is not None
                    ):
                        required = baseline_valid * (1.0 - float(cfg.early_stop_required_improvement))
                        if (best_valid is None) or (best_valid > required):
                            print(
                                f"[early-stop] best_valid={best_valid:.6f} baseline={baseline_valid:.6f} required<={required:.6f}. Stop."
                            )
                            passed = False
                            raise KeyboardInterrupt

                step += 1
                if step >= cfg.ntraining_steps:
                    break

            # --- エポック終端処理 ---
            epoch_avg = (
                epoch_train_loss / steps_per_epoch if steps_per_epoch > 0 else 0.0
            )
            train_loss_hist.append((epoch, float(epoch_avg)))

            if dl_valid is not None and validation_interval is not None:
                sampled_valid_example = next(iter(dl_valid))
                epoch_valid_loss = validation(
                    simulator, sampled_valid_example, n_features, cfg, device
                )
                valid_loss_hist.append((epoch, float(epoch_valid_loss.item())))
                print(
                    f"[epoch {epoch}] train={epoch_avg:.6f} valid={epoch_valid_loss.item():.6f}"
                )
            else:
                print(f"[epoch {epoch}] train={epoch_avg:.6f}")

            # リセット
            epoch_train_loss = 0.0
            steps_per_epoch = 0
            epoch += 1

            if step >= cfg.ntraining_steps:
                break

    except KeyboardInterrupt:
        print("Interrupted. Saving last state...")

    # 終了時に最終保存
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

    # 実験ログを CSV へ
    try:
        _append_exp_log(cfg, step, train_loss_hist, valid_loss_hist, passed)
    except Exception:
        pass


# -----------------------
# エントリーポイント
# -----------------------
def main():
    import argparse

    p = argparse.ArgumentParser(description="GNS Runner (no DDP)")
    p.add_argument("--config", "-c", default="config.yaml", help="Path to YAML config")
    p.add_argument("--mode", choices=["train", "valid", "rollout"], help="Override run mode", nargs="?")
    p.add_argument("--seed", type=int, help="Override RNG seed", nargs="?")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.mode is not None:
        cfg.mode = args.mode
    if args.seed is not None:
        cfg.seed = int(args.seed)

    # 乱数シード
    if cfg.seed is not None:
        try:
            import random, numpy as np

            random.seed(cfg.seed)
            np.random.seed(cfg.seed)
        except Exception:
            pass
        torch.manual_seed(cfg.seed)

    # デバイス選択
    if torch.cuda.is_available() and cfg.cuda_device_number is not None:
        device = torch.device(f"cuda:{int(cfg.cuda_device_number)}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")

    # 設定を保存(再現性のため)
    Path(cfg.model_path).mkdir(parents=True, exist_ok=True)
    with Path.open(
        Path(cfg.model_path) / "used_config.json", "w", encoding="utf-8"
    ) as f:
        json.dump(_asjsonable_cfg(cfg), f, ensure_ascii=False, indent=2)

    # ビルド用 cfg（モデル上書きに使用）
    global CFG_FOR_BUILD
    CFG_FOR_BUILD = cfg

    if cfg.mode == "train":
        train(cfg, device)
    elif cfg.mode in ("valid", "rollout"):
        predict(cfg, device)
    else:
        msg = f"Unknown mode: {cfg.mode}"
        raise ValueError(msg)


if __name__ == "__main__":
    main()

# -----------------------
# 付帯ユーティリティ
# -----------------------
def _append_exp_log(
    cfg: Config,
    step: int,
    train_hist: list[tuple[int, float]],
    valid_hist: list[tuple[int, float]],
    passed: bool | None,
):
    import csv
    import subprocess
    from datetime import datetime

    commit = "unknown"
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        pass

    exp_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    it_s = None
    # 最終 10 エポックの平均を簡易 it/s として保存（近似）
    # 計算済みの値がないので NA にしておく

    # 代表値抽出
    def _pick(hist: list[tuple[int, float]], epoch: int | None = None):
        if not hist:
            return None
        return float(hist[-1][1])

    train_at_end = _pick(train_hist)
    valid_at_end = _pick(valid_hist)

    # rollout20 は predict 時に評価する前提なのでここでは NA
    row = dict(
        exp_id=exp_id,
        git_commit=commit,
        cfg=str(cfg.__dict__),
        seed=cfg.seed,
        time_min=None,
        it_s=it_s,
        step=step,
        train_loss=train_at_end,
        valid_one_step=valid_at_end,
        rollout20_mse=None,
        passed=passed,
    )

    Path(cfg.model_path).mkdir(parents=True, exist_ok=True)
    fpath = Path(cfg.model_path) / "exp_log.csv"
    write_header = not fpath.exists()
    with fpath.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
