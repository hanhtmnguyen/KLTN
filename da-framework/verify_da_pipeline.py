"""
Verification script for the Supervised PI-GAN Data Assimilation (`da-framework`) repository.

Validates that all modules can be imported without error, checks syntax and class signatures,
and performs a quick forward-pass smoke test on the generator, discriminator, unsteady SWE loss,
and boundary condition data assimilation losses using synthetic tensors.
"""

from __future__ import annotations

import sys
from pathlib import Path
import torch
import numpy as np

# Ensure root is in path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging_config import get_logger, setup_logger

setup_logger("da", level=20)  # INFO level
logger = get_logger("da.verify")


def verify_imports():
    """Verify all package and module imports across the repository."""
    logger.info("=== 1. Verifying Module Imports ===")
    
    import src.utils.geo_utils as geo_utils
    import src.utils.visualization as viz
    
    import src.data_acquisition.fabdem_downloader as fabdem
    import src.data_acquisition.icesat2_atl03 as atl03
    import src.data_acquisition.sentinel2_downloader as s2
    import src.data_acquisition.icesat2_atl13 as atl13
    import src.data_acquisition.geoglows_discharge as geoglows
    import src.data_acquisition.sentinel1_sar as sar
    
    import src.preprocessing.ann_dtm_corrector as ann
    import src.preprocessing.terrain_features as terrain
    import src.preprocessing.roughness_matrix as roughness
    import src.preprocessing.osm_buildings as osm
    import src.preprocessing.hand_model as hand
    
    import src.groundtruth.hecras_runner as hecras
    import src.groundtruth.training_data_builder as builder
    
    import src.model.fourier_features as fourier
    import src.model.generator as generator
    import src.model.discriminator as discriminator
    import src.model.physics_loss as physics
    import src.model.boundary_condition_loss as bc_loss
    import src.model.data_assimilation as da_loss
    import src.model.dataset as dataset
    import src.model.trainer as trainer
    import src.model.inference as inference
    
    import src.assessment.metrics as metrics
    import src.assessment.scenario_runner as scenario
    import src.assessment.risk_mapper as risk
    
    logger.info("All 22 core packages and modules imported successfully!")


def verify_forward_pass_smoke_test():
    """Perform a synthetic forward pass through spatiotemporal generator and DA losses."""
    logger.info("=== 2. Running Forward Pass Smoke Test ===")
    
    from src.model.generator import PIGANGenerator
    from src.model.discriminator import PatchGANDiscriminator
    from src.model.physics_loss import UnsteadyPhysicsLoss
    from src.model.boundary_condition_loss import UpstreamBCLoss, DownstreamBCLoss
    from src.model.data_assimilation import UnifiedDataAssimilationLoss
    
    device = torch.device("cpu")
    B, C_phys, H, W = 2, 8, 32, 32
    
    # 1. Spatiotemporal Generator
    logger.info("Testing Spatiotemporal PIGANGenerator...")
    gen = PIGANGenerator(in_physical_channels=C_phys, out_channels=3, temporal=True).to(device)
    x_phys = torch.randn(B, C_phys, H, W, device=device)
    out_t0 = gen(x_phys, time_norm=0.0)
    out_t1 = gen(x_phys, time_norm=0.5)
    assert out_t0.shape == (B, 3, H, W), f"Expected shape (B,3,H,W), got {out_t0.shape}"
    assert out_t1.shape == (B, 3, H, W), f"Expected shape (B,3,H,W), got {out_t1.shape}"
    logger.info("  Generator forward pass verified (shapes: %s).", out_t0.shape)
    
    # 2. Discriminator
    logger.info("Testing PatchGANDiscriminator...")
    disc = PatchGANDiscriminator(in_channels=3).to(device)
    disc_out = disc(out_t0)
    logger.info("  Discriminator forward pass verified (shape: %s).", disc_out.shape)
    
    # 3. Unsteady SWE Physics Loss
    logger.info("Testing UnsteadyPhysicsLoss...")
    unsteady_loss = UnsteadyPhysicsLoss(gravity=9.81, dx=10.0, dy=10.0, dt=3600.0).to(device)
    h0, u0, v0 = out_t0[:, 0:1], out_t0[:, 1:2], out_t0[:, 2:3]
    h1, u1, v1 = out_t1[:, 0:1], out_t1[:, 1:2], out_t1[:, 2:3]
    z_b = torch.rand(B, 1, H, W, device=device) * 15.0
    n_manning = torch.full((B, 1, H, W), 0.035, device=device)
    l_phys = unsteady_loss(h0, u0, v0, h1, u1, v1, z_b, n_manning)
    assert not torch.isnan(l_phys), "Unsteady physics loss returned NaN!"
    logger.info("  Unsteady SWE loss evaluated successfully: %.4f", l_phys.item())
    
    # 4. Boundary Condition Losses (GEOGloWS & ATL13)
    logger.info("Testing Upstream & Downstream Boundary Condition Losses...")
    up_mask = torch.zeros((B, H, W), device=device)
    up_mask[:, :, 0] = 1.0  # Leftmost column as upstream boundary
    q_target = torch.tensor([2500.0, 3200.0], device=device)
    
    up_bc = UpstreamBCLoss(cell_width=10.0)
    l_up = up_bc(h1, u1, v1, up_mask, q_target)
    
    down_bc = DownstreamBCLoss()
    # Simulated ATL13 points [row, col, wse]
    obs_batch = [
        torch.tensor([[5.0, 15.0, 12.5], [10.0, 20.0, 11.8]], device=device),
        torch.tensor([[8.0, 12.0, 13.0]], device=device)
    ]
    l_down = down_bc(h1, z_b, obs_batch)
    
    logger.info("  L_upstream evaluated successfully: %.4f", l_up.item())
    logger.info("  L_downstream evaluated successfully: %.4f", l_down.item())
    
    # 5. Unified DA Loss
    logger.info("Testing UnifiedDataAssimilationLoss...")
    unified_da = UnifiedDataAssimilationLoss().to(device)
    sar_mask = torch.randint(0, 2, (B, 1, H, W), device=device).float()
    da_dict = unified_da(h1, u1, v1, z_b, up_mask, q_target, obs_batch, sar_mask)
    assert not torch.isnan(da_dict["total"]), "Unified DA loss returned NaN!"
    logger.info("  Unified DA total loss evaluated successfully: %.4f", da_dict["total"].item())
    
    logger.info("=== All Smoke Tests Passed Successfully! ===")


if __name__ == "__main__":
    try:
        verify_imports()
        verify_forward_pass_smoke_test()
    except Exception as e:
        logger.error("Verification failed: %s", e, exc_info=True)
        sys.exit(1)
