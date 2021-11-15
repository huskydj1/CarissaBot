import torch
import torch.nn as nn

class SE_Block(nn.Module):
    def __init__(self, filters, se_channels):
        super().__init__()
        self.filters = filters
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(filters, se_channels),
            nn.ReLU(inplace=True),
            nn.Linear(se_channels, 2*filters),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        w = y[:, 0:self.filters, :, :]
        b = y[:, self.filters:2*self.filters, :, :]
        w = torch.sigmoid(w)
        return x * w.expand_as(x) + b.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, filters, se_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(filters)
        self.se = SE_Block(filters, se_channels)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out = out + x
        out = torch.relu(out)
        return out

class CarissaNet(nn.Module):
    def __init__(self, blocks=20, filters=256, se_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(29, filters, 3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(filters)
        
        self.residual_blocks = nn.ModuleList([ResidualBlock(filters, se_channels) for _ in range(blocks)])

        self.conv2 = nn.Conv2d(filters, 32, 3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Sequential(
            nn.Linear(32*8*8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)

        for block in self.residual_blocks:
            out = block(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp

    net = CarissaNet(blocks=10, filters=128, se_channels=32)
    print(net)
    print(get_n_params(net))