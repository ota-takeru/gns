import torch

# 基本の加速度MSEロス（元の実装と同一ロジック）
def acceleration_loss(
    pred_acc: torch.Tensor,
    target_acc: torch.Tensor,
    non_kinematic_mask: torch.Tensor,
) -> torch.Tensor:
    """非運動学粒子のみを対象にした加速度MSE"""
    loss = (pred_acc - target_acc) ** 2
    loss = loss.sum(dim=-1)
    num_non_kinematic = non_kinematic_mask.sum()
    masked = torch.where(
        non_kinematic_mask.bool(), loss, torch.zeros_like(loss)
    )
    return masked.sum() / num_non_kinematic.clamp(min=1)


LOSS_REGISTRY = {
    "acceleration": acceleration_loss,
}


def get_loss(name: str = "acceleration"):
    try:
        return LOSS_REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - defensive
        known = ", ".join(sorted(LOSS_REGISTRY))
        raise ValueError(f"Unknown loss '{name}'. known: {known}") from exc
