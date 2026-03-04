import torch
import pytest
from src.losses.topology import SoftclDiceLoss, soft_skel
from src.losses import build_loss, TopologyAwareLoss

def test_soft_skel():
    """Smoke test for soft skeletonization."""
    img = torch.rand((2, 1, 64, 64))
    skel = soft_skel(img, iters=3)
    assert skel.shape == img.shape
    assert not torch.isnan(skel).any()
    assert skel.min() >= 0

def test_soft_cldice_loss():
    """Smoke test for SoftclDiceLoss."""
    criterion = SoftclDiceLoss(iters=3)
    pred = torch.randn((2, 1, 64, 64), requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    
    loss = criterion(pred, target)
    assert loss.item() >= 0
    loss.backward()
    assert pred.grad is not None

def test_topology_aware_loss():
    """Smoke test for combined TopologyAwareLoss."""
    criterion = TopologyAwareLoss(cldice_weight=0.5)
    pred = torch.randn((2, 1, 64, 64), requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    
    loss = criterion(pred, target)
    assert loss.item() >= 0
    loss.backward()
    assert pred.grad is not None

def test_build_loss_topology():
    """Test build_loss factory with topology_aware."""
    cfg = {
        "loss": "topology_aware",
        "cldice_weight": 0.1,
        "cldice_iters": 5
    }
    criterion = build_loss(cfg)
    assert isinstance(criterion, TopologyAwareLoss)
    assert criterion.cldice_weight == 0.1
    assert criterion.cldice.iters == 5
