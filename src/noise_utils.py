import torch

import learned_simulator


# position列を受け取ってランダムウォークノイズを生成する。
# 具体的には、速度に関するランダムウォークノイズを生成し、位置のノイズに変換する。
def get_random_walk_noise_for_position_sequence(
    position_sequence: torch.Tensor, noise_std_last_step: float
) -> torch.Tensor:  # position_sequence: [N, T, D]
    velocity_sequence = learned_simulator.time_diff(position_sequence)  # 速度列を計算

    num_velocities = velocity_sequence.shape[1]
    if num_velocities == 0:
        return torch.zeros_like(position_sequence)

    noise_scale = noise_std_last_step / float(num_velocities) ** 0.5
    velocity_sequence_noise = torch.randn_like(
        velocity_sequence, device=position_sequence.device
    ) * noise_scale  # 最後のステップでの標準偏差に合わせて調整
    velocity_sequence_noise = torch.cumsum(
        velocity_sequence_noise, dim=1
    )  # 速度ノイズの累積和を取ってランダムウォークノイズを生成

    position_sequence_noise = torch.zeros_like(
        position_sequence
    )  # 最初の位置はノイズなし
    position_sequence_noise[:, 1:] = torch.cumsum(
        velocity_sequence_noise, dim=1
    )  # 速度の累積和は位置
    return position_sequence_noise


NOISE_REGISTRY = {
    "random_walk": get_random_walk_noise_for_position_sequence,
}


def get_noise(name: str = "random_walk"):
    try:
        return NOISE_REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - defensive
        known = ", ".join(sorted(NOISE_REGISTRY))
        raise ValueError(f"Unknown noise '{name}'. known: {known}") from exc
