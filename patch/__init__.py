from patch.trainer import apply_patch as apply_trainer_patch
from patch.l2warp import apply_patch as apply_l2warp_patch

def apply_patches():
    apply_trainer_patch()
    apply_l2warp_patch()