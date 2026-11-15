"""Wide ResNet WRN-depth-k (Zagoruyko & Komodakis, paper ref [42]); paper uses WRN-28-4 on CIFAR-100.

depth=28 -> 4 basic blocks per stage, widening factor 4 -> stage widths 64/128/256, BatchNorm in blocks.
ASSUMPTION: dropout 0 (paper does not mention dropout), Kaiming-normal conv init (standard WRN practice).
"""
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.bn1, self.conv1 = nn.BatchNorm2d(cin), nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2, self.conv2 = nn.BatchNorm2d(cout), nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.short = None if (cin == cout and stride == 1) else nn.Conv2d(cin, cout, 1, stride, 0, bias=False)

    def forward(self, x):
        o = F.relu(self.bn1(x))
        y = self.conv1(o)
        y = self.conv2(F.relu(self.bn2(y)))
        return y + (x if self.short is None else self.short(o))


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen=4, num_classes=100):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        w = [16, 16 * widen, 32 * widen, 64 * widen]
        self.conv1 = nn.Conv2d(3, w[0], 3, 1, 1, bias=False)
        blocks, cin = [], w[0]
        for i, stride in enumerate([1, 2, 2]):
            for j in range(n):
                blocks.append(_Block(cin, w[i + 1], stride if j == 0 else 1))
                cin = w[i + 1]
        self.blocks = nn.Sequential(*blocks)
        self.bn, self.fc = nn.BatchNorm2d(cin), nn.Linear(cin, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.bn(self.blocks(self.conv1(x))))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


def wrn28_4(num_classes=100):
    return WideResNet(28, 4, num_classes)
