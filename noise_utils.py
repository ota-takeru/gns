import Learned_simulator
import torch

#position列を受け取ってランダムウォークノイズを生成する。
#具体的には、速度に関するランダムウォークノイズを生成し、位置のノイズに変換する。
def get_random_walk_noise_for_position_sequence(
        position_sequence: torch.tensor,
        noise_std_last_step: float) -> torch.tensor:
    velocity_sequence = Learned_simulator.time_diff(position_sequence) #速度列を計算

    num_velocities = velocity_sequence.shape[1]
    velocity_sequence_noise = torch.randn(
      list(velocity_sequence.shape)) * (noise_std_last_step/num_velocities**0.5) #最後のステップでの標準偏差に合わせて調整
    velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1) #速度ノイズの累積和を取ってランダムウォークノイズを生成

    position_sequence_noise = torch.zeros_like(position_sequence) #最初の位置はノイズなし
    position_sequence_noise[:, 1:] = torch.cumsum(velocity_sequence_noise, dim=1) #速度の累積和は位置
    return position_sequence_noise
