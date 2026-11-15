"""CNNs for the three datasets (paper Sec. VI).

Paper facts: EMNIST CNN = two 5x5 conv + two FC layers (47 outputs); CIFAR-10 CNN = two 5x5
padded conv layers; CIFAR-100 CNN = three 3x3 padded conv layers + max pooling + two FC (100 outputs).
ASSUMPTIONS (not given in the paper): channel widths 32/64(/128), ReLU, 2x2 max-pool after every conv,
hidden FC width 512, no dropout/BatchNorm, PyTorch default init; for CIFAR-10 the FC head is
two FC layers (as in the EMNIST CNN).
"""
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, in_ch: int, spatial: int, conv_cfg, num_classes: int, hidden: int = 512):
        super().__init__()
        layers, ch = [], in_ch
        for out_ch, k in conv_cfg:
            layers += [nn.Conv2d(ch, out_ch, k, padding=k // 2), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            ch, spatial = out_ch, spatial // 2
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(ch * spatial * spatial, hidden),
                                        nn.ReLU(inplace=True), nn.Linear(hidden, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def emnist_cnn():    return SimpleCNN(1, 28, [(32, 5), (64, 5)], 47)
def cifar10_cnn():   return SimpleCNN(3, 32, [(32, 5), (64, 5)], 10)
def cifar100_cnn():  return SimpleCNN(3, 32, [(32, 3), (64, 3), (128, 3)], 100)
