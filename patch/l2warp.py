import torch
from fla.modules.l2warp import L2Wrap

def backward(ctx, grad_output):
    maxx, ids = ctx.saved_tensors
    glogits = torch.zeros(ctx.logits_shape, device=grad_output.device,
                            dtype=maxx.dtype)
    glogits.scatter_(-1, ids, maxx)
    return grad_output, glogits, None

def apply_patch():
    L2Wrap.backward = staticmethod(backward)