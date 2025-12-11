import torch

from learned_simulator import GNSSimulator
from losses import acceleration_loss


def _build_dummy_sim():
    stats = {
        "acceleration": {"mean": torch.zeros(2), "std": torch.ones(2)},
        "velocity": {"mean": torch.zeros(2), "std": torch.ones(2)},
    }
    boundaries = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    # node feature dim = velocity(5*2) + boundary(2*2) + type_emb(4) = 18
    return GNSSimulator(
        particle_dimensions=2,
        nnode_in=18,
        nedge_in=3,
        latent_dim=16,
        nmessage_passing_steps=2,
        nmlp_layers=2,
        mlp_hidden_dim=32,
        connectivity_radius=0.4,
        normalization_stats=stats,
        nparticle_types=4,
        particle_type_embedding_size=4,
        boundaries=boundaries,
        boundary_clamp_limit=1.0,
        device="cpu",
    )


def test_predict_positions_smoke():
    torch.manual_seed(0)
    sim = _build_dummy_sim()
    positions = torch.randn(4, 6, 2)
    particle_types = torch.zeros(4, dtype=torch.long)
    nparticles = torch.tensor([4])

    out = sim.predict_positions(
        positions, nparticles_per_example=nparticles, particle_types=particle_types
    )
    assert out.shape == (4, 2)


def test_acceleration_loss_masks_kinematic():
    pred = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.zeros_like(pred)
    non_kinematic = torch.tensor([True, False])
    loss = acceleration_loss(pred, target, non_kinematic)
    # 2つ目はマスクされるので1粒子ぶんだけが残る
    assert torch.isclose(loss, torch.tensor(1.0))
