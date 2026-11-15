"""Server side: global model evaluation and BatchNorm-buffer averaging."""
from typing import Dict, List

import torch
import torch.nn.functional as F


def average_buffers(buf_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """ASSUMPTION: non-parameter buffers (BN running stats, WRN only) are averaged across the S clients;
    DRAG itself is defined on trainable parameters only."""
    out = {}
    for k in buf_list[0]:
        t = torch.stack([b[k] for b in buf_list])
        out[k] = t.float().mean(0).to(t.dtype) if t.is_floating_point() else t[0].clone()
    return out


@torch.no_grad()
def evaluate(model, x, y, device, batch_size=1000):
    """Global-model loss/accuracy on the held-out TEST set."""
    model.eval()
    tot_loss, correct = 0.0, 0
    for i in range(0, len(x), batch_size):
        xb, yb = x[i:i + batch_size].to(device), y[i:i + batch_size].to(device)
        out = model(xb)
        tot_loss += F.cross_entropy(out, yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item()
    return tot_loss / len(x), 100.0 * correct / len(x)
