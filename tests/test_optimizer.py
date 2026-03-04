import torch
from src.models.swinunetr_vlm_v1 import SwinUNETR2DVLMV1
from src.utils.exp import build_optimizer

def test_lr_split():
    """Verify that build_optimizer correctly splits parameters and assigns different LRs."""
    model = SwinUNETR2DVLMV1(vlm_mode="both")
    cfg = {
        "model": "swinunetr_vlm_v1",
        "lr": 1e-4,
        "lr_injection": 1e-3,
        "weight_decay": 1e-5
    }
    
    optimizer = build_optimizer(model, cfg)
    
    assert len(optimizer.param_groups) == 2
    
    # Identify which group is which
    # Usually injection_params are in the second group if injection_params exists
    lrs = [group['lr'] for group in optimizer.param_groups]
    assert 1e-4 in lrs
    assert 1e-3 in lrs
    
    # Check if correct params are in the higher LR group
    injection_group = next(g for g in optimizer.param_groups if g['lr'] == 1e-3)
    base_group = next(g for g in optimizer.param_groups if g['lr'] == 1e-4)
    
    # Sampling parameters
    injection_param_names = [n for n, p in model.named_parameters() 
                             if any(n.startswith(pre) for pre in ("text_gates", "img_injectors"))]
    
    # Check that at least one injection parameter is in the injection_group
    # Note: parameters in param_groups are the tensors themselves, not names.
    # We can check by id.
    injection_ids = [id(p) for p in injection_group['params']]
    
    # Find an img_injector parameter
    example_param = model.img_injectors[0].alpha
    assert id(example_param) in injection_ids
    
    # Find a base swin parameter
    # SwinUNETR.out is a UnetOutBlock containing a Convolution which contains a Conv2d
    swin_param = model.swin.out.conv.conv.weight
    assert id(swin_param) in [id(p) for p in base_group['params']]

def test_no_split_for_baseline():
    """Verify that baseline models (non-VLM) only have one param group."""
    from src.models.swinunetr import SwinUNETR2D
    model = SwinUNETR2D()
    cfg = {
        "model": "swinunetr",
        "lr": 1e-4,
        "weight_decay": 1e-5
    }
    
    optimizer = build_optimizer(model, cfg)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]['lr'] == 1e-4
