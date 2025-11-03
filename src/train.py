import json
import math
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import data_loader
from scenarios import Scenario, ScenarioRegistry

# --- gns モジュール群(あなたの環境のパスに応じて調整してください) ---
import learned_simulator
import noise_utils
import reading_utils

try:
    from tensorboard import program as tensorboard_program
except ImportError:  # pragma: no cover - optional dependency
    tensorboard_program = None

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
    scenario: str = "rigid"
    scenario_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_scenario: Scenario | None = field(default=None, repr=False)
    model_path: str = "./models/"
    output_path: str = "./rollouts/"
    output_filename: str = "rollout"

    batch_size: int = 2
    noise_std: float = 6.7e-4
    log_interval: int = 10

    ntraining_steps: int = int(2e7)
    validation_interval: int | None = None
    nsave_steps: int = 5000
    rollout_interval: int | None = None
    rollout_max_examples: int = 1
    rollout_dataset: str | None = None
    # rollout(推論)時に保存するシーン数の上限(Noneで全件)
    rollout_inference_max_examples: int | None = None
    train_dataset_fraction: float | None = None
    valid_dataset_fraction: float | None = None
    train_dataset_count: int | None = None
    valid_dataset_count: int | None = None
    tensorboard_interval: int | None = None

    lr_init: float = 1e-4
    lr_decay: float = 0.1
    lr_decay_steps: int = int(5e6)

    model_file: str | None = None  # "latest" (inference default) or ファイル名 or null
    train_state_file: str | None = "train_state.pt"  # "latest" or ファイル名 or null

    cuda_device_number: int | None = None  # null で自動選択
    tensorboard_log_dir: str | None = None
    tensorboard_port: int | None = None
    tensorboard_host: str | None = None


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
        "scenario": cfg.scenario,
        "scenario_options": cfg.scenario_options,
        "active_scenario": (
            {
                "key": cfg.active_scenario.key,
                "dataset_dir": str(cfg.active_scenario.dataset_dir),
            }
            if cfg.active_scenario
            else None
        ),
        "model_path": cfg.model_path,
        "output_path": cfg.output_path,
        "output_filename": cfg.output_filename,
        "batch_size": cfg.batch_size,
        "noise_std": cfg.noise_std,
        "log_interval": cfg.log_interval,
        "ntraining_steps": cfg.ntraining_steps,
        "validation_interval": cfg.validation_interval,
        "nsave_steps": cfg.nsave_steps,
        "rollout_interval": cfg.rollout_interval,
        "rollout_max_examples": cfg.rollout_max_examples,
        "rollout_dataset": cfg.rollout_dataset,
        "rollout_inference_max_examples": cfg.rollout_inference_max_examples,
        "train_dataset_fraction": cfg.train_dataset_fraction,
        "valid_dataset_fraction": cfg.valid_dataset_fraction,
        "train_dataset_count": cfg.train_dataset_count,
        "valid_dataset_count": cfg.valid_dataset_count,
        "tensorboard_interval": cfg.tensorboard_interval,
        "lr_init": cfg.lr_init,
        "lr_decay": cfg.lr_decay,
        "lr_decay_steps": cfg.lr_decay_steps,
        "model_file": cfg.model_file,
        "train_state_file": cfg.train_state_file,
        "cuda_device_number": cfg.cuda_device_number,
        "tensorboard_log_dir": cfg.tensorboard_log_dir,
        "tensorboard_port": cfg.tensorboard_port,
        "tensorboard_host": cfg.tensorboard_host,
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


def _launch_tensorboard(
    log_dir: Path, host: str | None, port: int | None
) -> tuple[Any | None, str | None]:
    if tensorboard_program is None:
        return None, None
    tb = tensorboard_program.TensorBoard()
    argv = [None, "--logdir", str(log_dir)]
    if host:
        argv += ["--host", host]
    if port:
        argv += ["--port", str(port)]
    try:
        tb.configure(argv=argv)
        url = tb.launch()
    except Exception:  # pragma: no cover - runtime environment dependent
        return None, None
    return tb, url


class RolloutEvaluator:
    """Utility to compute rollout stability metrics on demand."""

    def __init__(
        self,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
        max_examples: int = 1,
    ):
        self._loader = loader
        self._device = device
        self._max_examples = max(1, int(max_examples))
        self._iterator: Any | None = None

    def evaluate(
        self, simulator: learned_simulator.LearnedSimulator
    ) -> dict[str, float | None]:
        metrics_list: list[dict[str, float]] = []
        for _ in range(self._max_examples):
            example = self._next_example()
            if example is None:
                break
            metrics = self._evaluate_example(simulator, example)
            if metrics is not None:
                metrics_list.append(metrics)

        if not metrics_list:
            return {
                "rollout_rmse_mean": None,
                "rollout_rmse_last": None,
                "rollout_instability": None,
            }

        aggregated: dict[str, float] = {}
        for key in metrics_list[0]:
            aggregated[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
        return aggregated

    def _next_example(self):
        if self._iterator is None:
            self._iterator = iter(self._loader)
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._loader)
            try:
                return next(self._iterator)
            except StopIteration:
                return None

    def _evaluate_example(
        self, simulator: learned_simulator.LearnedSimulator, example: tuple
    ) -> dict[str, float] | None:
        if len(example) == 4:
            positions, particle_type, material_property, n_particles = example
            material_property_tensor = material_property.to(self._device)
            n_particles_tensor = torch.tensor(
                [int(n_particles)], dtype=torch.int32, device=self._device
            )
        elif len(example) == 3:
            positions, particle_type, n_particles = example
            material_property_tensor = None
            n_particles_tensor = torch.tensor(
                [int(n_particles)], dtype=torch.int32, device=self._device
            )
        else:
            return None

        positions = positions.to(self._device)
        particle_type = particle_type.to(self._device)

        nsteps = positions.shape[1] - INPUT_SEQUENCE_LENGTH
        if nsteps <= 0:
            return None

        _, loss = rollout(
            simulator,
            positions,
            particle_type,
            material_property_tensor,
            n_particles_tensor,
            nsteps,
            self._device,
            show_progress=False,
        )

        if loss.numel() == 0:
            return None

        rmse_per_step = torch.sqrt(loss.mean(dim=(1, 2)))
        rmse_mean = float(rmse_per_step.mean().item())
        rmse_last = float(rmse_per_step[-1].item())
        if rmse_mean > 0.0:
            instability = float(rmse_last / rmse_mean)
        else:
            instability = 0.0

        return {
            "rollout_rmse_mean": rmse_mean,
            "rollout_rmse_last": rmse_last,
            "rollout_instability": instability,
        }


def _compute_grad_norm(parameters) -> float | None:
    total = 0.0
    has_grad = False
    for param in parameters:
        if param.grad is None:
            continue
        total += float(param.grad.detach().pow(2).sum().item())
        has_grad = True
    if not has_grad:
        return None
    return math.sqrt(total)


def _resolve_rollout_dataset_path(cfg: Config) -> Path | None:
    base = Path(cfg.data_path)
    if cfg.rollout_dataset:
        candidate = base / f"{cfg.rollout_dataset}.npz"
        if candidate.exists():
            return candidate
    scenario = cfg.active_scenario
    if scenario and scenario.rollout_dataset:
        candidate = base / f"{scenario.rollout_dataset}.npz"
        if candidate.exists():
            return candidate
    for split_name in ("valid", "test", "train"):
        candidate = base / f"{split_name}.npz"
        if candidate.exists():
            return candidate
    return None


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

    if "bounds" not in metadata:
        msg = (
            "Dataset metadata is missing 'bounds'. "
            "Regenerate the dataset with wall-distance features."
        )
        raise KeyError(msg)
    boundaries = np.asarray(metadata["bounds"], dtype=np.float32)
    if boundaries.ndim != 2 or boundaries.shape[1] != 2:
        msg = (
            f"Invalid bounds shape {boundaries.shape}, expected (dim, 2). "
            "Check dataset generation."
        )
        raise ValueError(msg)
    boundary_clamp_limit = float(metadata.get("boundary_augment", 1.0))

    simulator = learned_simulator.LearnedSimulator(
        particle_dimensions=metadata["dim"],
        nnode_in=nnode_in,
        nedge_in=nedge_in,
        latent_dim=128,
        nmessage_passing_steps=10,
        nmlp_layers=2,
        mlp_hidden_dim=128,
        connectivity_radius=metadata["default_connectivity_radius"],
        normalization_stats=normalization_stats,
        nparticle_types=NUM_PARTICLE_TYPES,
        particle_type_embedding_size=16,
        boundaries=boundaries,
        boundary_clamp_limit=boundary_clamp_limit,
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
    *,
    show_progress: bool = True,
) -> tuple[dict[str, Any], torch.Tensor]:
    """逐次1ステップ予測をシフト窓で積み上げるロールアウト
    position: (B, T, N, D) を想定。ここでは B=1 を前提に扱っている実装。
    """
    # B 次元を想定しつつコードは元実装を踏襲(B=1前提)
    initial_positions = position[:, :INPUT_SEQUENCE_LENGTH]
    ground_truth_positions = position[:, INPUT_SEQUENCE_LENGTH:]

    current_positions = initial_positions
    predictions_list = []

    step_iterator = range(nsteps)
    if show_progress:
        step_iterator = tqdm(step_iterator, total=nsteps)

    for step_idx in step_iterator:
        next_position = simulator.predict_positions(
            current_positions,
            nparticles_per_example=n_particles_per_example,
            particle_types=particle_types,
            material_property=material_property,
        )
        # 運動学粒子は教師で上書き
        kinematic_mask = (particle_types == KINEMATIC_PARTICLE_ID).to(device)
        next_position_gt = ground_truth_positions[:, step_idx]  # (B=1, N, D)
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
    metadata_key = (
        cfg.active_scenario.rollout_metadata_split
        if cfg.active_scenario and cfg.active_scenario.rollout_metadata_split
        else "rollout"
    )
    metadata = reading_utils.read_metadata(cfg.data_path, metadata_key)
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
        path=Path(cfg.data_path) / f"{split}.npz"
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
            # 推論で生成・可視化するシーン数の上限が設定されていれば適用
            if (
                cfg.mode == "rollout"
                and cfg.rollout_inference_max_examples is not None
                and example_i >= int(cfg.rollout_inference_max_examples)
            ):
                print(
                    f"Reached rollout_inference_max_examples={cfg.rollout_inference_max_examples}. Stopping."
                )
                break
            print(f"processing example number {example_i}")
            positions = features[0].to(device)
            # nsteps 計算
            if metadata.get("sequence_length") is not None:
                nsteps = int(metadata["sequence_length"]) - INPUT_SEQUENCE_LENGTH
            else:
                sequence_length = positions.shape[1]
                nsteps = int(sequence_length) - INPUT_SEQUENCE_LENGTH

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


def _prepare_model_directory(cfg: Config) -> None:
    """Ensure model outputs are written to an isolated run directory."""
    base_dir = Path(cfg.model_path).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    # Stash base directory for later lookup (e.g., inference latest resolution).
    setattr(cfg, "model_base_path", str(base_dir))

    if cfg.mode == "train" and cfg.model_file is None:
        timestamp = time.strftime("run-%Y%m%d-%H%M%S")
        run_dir = base_dir / timestamp
        counter = 1
        while run_dir.exists():
            run_dir = base_dir / f"{timestamp}-{counter:02d}"
            counter += 1
        run_dir.mkdir(parents=True, exist_ok=False)
        run_dir = run_dir.resolve()
        setattr(cfg, "model_run_path", str(run_dir))
        cfg.model_path = str(run_dir)
    else:
        setattr(cfg, "model_run_path", None)
        cfg.model_path = str(base_dir)


def _resolve_model_path(cfg: Config) -> str:
    """cfg.model_file が 'latest' のとき最新を解決。そうでなければ結合して返す。"""
    Path(cfg.model_path).mkdir(parents=True, exist_ok=True)
    model_file = cfg.model_file
    if model_file is None and cfg.mode in ("valid", "rollout"):
        model_file = "latest"

    base_dir = Path(getattr(cfg, "model_base_path", cfg.model_path)).expanduser().resolve()
    search_root = base_dir if base_dir.is_dir() else base_dir.parent

    if model_file == "latest":
        expr = re.compile(r"model-(\d+)\.pt$")
        candidates: list[tuple[float, int, Path]] = []
        for path in search_root.rglob("model-*.pt"):
            match = expr.search(path.name)
            if not match:
                continue
            try:
                step_num = int(match.group(1))
            except ValueError:
                step_num = -1
            stat = path.stat()
            candidates.append((stat.st_mtime, step_num, path.resolve()))
        if not candidates:
            msg = f"No model files found in {search_root}"
            raise FileNotFoundError(msg)
        candidates.sort()
        return str(candidates[-1][2])
    if model_file:
        candidate = Path(model_file)
        if candidate.is_file():
            return str(candidate.resolve())

        search_roots = [Path(cfg.model_path).expanduser().resolve()]
        if base_dir not in search_roots:
            search_roots.append(base_dir)

        for root in search_roots:
            resolved = (root / model_file).expanduser().resolve()
            if resolved.exists():
                return str(resolved)

        msg = f"Model file {model_file} not found under {', '.join(str(r) for r in search_roots)}"
        raise FileNotFoundError(msg)
    # 明示されていない場合新規学習用なので存在しないパスを返す
    return str(Path(cfg.model_path) / "model-init.pt")


# -----------------------
# 学習
# -----------------------
def train(cfg: Config, device: torch.device):
    # メタ情報
    metadata_key = (
        cfg.active_scenario.metadata_split if cfg.active_scenario else "train"
    )
    metadata = reading_utils.read_metadata(cfg.data_path, metadata_key)
    train_loss = 0.0  # 初期化

    # モデルと最適化器
    simulator = _get_simulator(metadata, cfg.noise_std, cfg.noise_std, device)
    simulator.to(device)
    optimizer = torch.optim.Adam(simulator.parameters(), lr=cfg.lr_init)
    current_lr = cfg.lr_init

    log_interval = max(1, int(cfg.log_interval))
    if cfg.tensorboard_interval is not None:
        tensorboard_interval = max(1, int(cfg.tensorboard_interval))
    else:
        tensorboard_interval = log_interval

    tb_writer: SummaryWriter | None = None
    tb_server: Any | None = None
    tb_url: str | None = None
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
    except Exception as exc:  # pragma: no cover - optional dependency/path issues
        print(f"Warning: could not initialize TensorBoard writer: {exc}")
        tb_writer = None
        tb_server = None

    # 進捗状態
    step = 0
    epoch = 0
    steps_per_epoch = 0

    valid_loss = None
    epoch_train_loss = 0.0
    epoch_valid_loss = None
    latest_valid_loss_value: float | None = None
    latest_rollout_metrics: dict[str, float | None] | None = None
    last_grad_norm: float | None = None
    last_logged_step = -1

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
        fraction=cfg.train_dataset_fraction,
        max_trajectories=cfg.train_dataset_count,
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
            fraction=cfg.valid_dataset_fraction,
            max_trajectories=cfg.valid_dataset_count,
            shuffle=False,
        )
        if len(dl_valid.dataset[0]) != n_features:
            msg = "`valid.npz` と `train.npz` の特徴数が一致していません。"
            raise ValueError(msg)

    rollout_evaluator: RolloutEvaluator | None = None
    rollout_interval = (
        int(cfg.rollout_interval)
        if cfg.rollout_interval and cfg.rollout_interval > 0
        else None
    )
    if rollout_interval is not None:
        rollout_dataset_path = _resolve_rollout_dataset_path(cfg)
        if rollout_dataset_path is None:
            print(
                "Warning: rollout metrics enabled but no suitable dataset was found; "
                "disabling rollout evaluation."
            )
            rollout_interval = None
        else:
            rollout_loader = data_loader.get_data_loader_by_trajectories(
                rollout_dataset_path
            )
            rollout_evaluator = RolloutEvaluator(
                rollout_loader, device, cfg.rollout_max_examples
            )
            print(
                f"Rollout metrics every {rollout_interval} steps using {rollout_dataset_path}"
            )

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
                    latest_valid_loss_value = float(valid_loss.item())
                    print(f"[valid @ step {step}] {latest_valid_loss_value:.6f}")
                    if tb_writer is not None:
                        tb_writer.add_scalar(
                            "valid/loss", latest_valid_loss_value, step
                        )

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

                # ロス → 逆伝播 → 更新
                loss = acceleration_loss(pred_acc, target_acc, non_kinematic_mask)
                train_loss = float(loss.item())
                epoch_train_loss += train_loss

                optimizer.zero_grad()
                loss.backward()
                grad_norm = _compute_grad_norm(simulator.parameters())
                last_grad_norm = grad_norm
                optimizer.step()

                # 学習率を指数減衰で更新
                lr_new = cfg.lr_init * (cfg.lr_decay ** (step / cfg.lr_decay_steps))
                for g in optimizer.param_groups:
                    g["lr"] = lr_new
                current_lr = lr_new

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
                    log_parts = [
                        f"epoch={epoch}",
                        f"step={step}/{cfg.ntraining_steps}",
                        f"loss={train_loss:.6f}",
                        f"lr={lr_new:.6e}",
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

                if tb_writer is not None and step % tensorboard_interval == 0:
                    tb_writer.add_scalar("train/loss", train_loss, step)
                    tb_writer.add_scalar("train/lr", lr_new, step)
                    if last_grad_norm is not None:
                        tb_writer.add_scalar("train/grad_norm", last_grad_norm, step)
                    if latest_valid_loss_value is not None:
                        tb_writer.add_scalar(
                            "valid/loss", latest_valid_loss_value, step
                        )
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
                    last_logged_step = step

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
                print(f"[epoch {epoch}] train={epoch_avg:.6f}")
            if tb_writer is not None:
                tb_writer.add_scalar("epoch/train_loss", epoch_avg, epoch)
                tb_writer.flush()

            # リセット
            epoch_train_loss = 0.0
            steps_per_epoch = 0
            epoch += 1

            if step >= cfg.ntraining_steps:
                break

    except KeyboardInterrupt:
        print("Interrupted. Saving last state...")
    finally:
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
            if last_logged_step != step:
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
            tb_writer.close()
        if (
            tb_server is not None
            and hasattr(tb_server, "_server")
            and tb_server._server is not None
        ):
            try:  # pragma: no cover - server cleanup
                tb_server._server.stop()
            except Exception:
                pass


# -----------------------
# エントリーポイント
# -----------------------
def main():
    import argparse

    p = argparse.ArgumentParser(description="GNS Runner (no DDP)")
    p.add_argument("--config", "-c", default="config.yaml", help="Path to YAML config")
    args = p.parse_args()

    cfg = load_config(args.config)
    base_dataset = Path(cfg.data_path).expanduser().resolve()
    registry = ScenarioRegistry(base_dataset, cfg.scenario_options)
    scenario = registry.get(cfg.scenario)
    scenario.apply_overrides(cfg)
    cfg.data_path = str(scenario.dataset_dir)
    if cfg.rollout_dataset is None and scenario.rollout_dataset:
        cfg.rollout_dataset = scenario.rollout_dataset
    cfg.active_scenario = scenario
    print(f"scenario: {scenario.key} @ {scenario.dataset_dir}")
    if scenario.description:
        print(f"  description: {scenario.description}")

    _prepare_model_directory(cfg)
    run_dir = getattr(cfg, "model_run_path", None)
    if run_dir:
        print(f"model run directory: {run_dir}")

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

    if cfg.mode == "train":
        train(cfg, device)
    elif cfg.mode in ("valid", "rollout"):
        predict(cfg, device)
    else:
        msg = f"Unknown mode: {cfg.mode}"
        raise ValueError(msg)


if __name__ == "__main__":
    main()
