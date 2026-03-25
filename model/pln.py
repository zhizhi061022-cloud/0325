"""
PLN model built on top of ResNet-18.

This implementation follows the paper's Figure 2 more closely:

    backbone -> shared conv stack -> 4 branches
    each branch: conv -> dilation stack -> inference
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


def _cbr(ic, oc, k=3, s=1, p=1, d=1):
    return nn.Sequential(
        nn.Conv2d(ic, oc, k, s, p, dilation=d, bias=False),
        nn.BatchNorm2d(oc),
        nn.ReLU(inplace=True),
    )


class ResNet18Backbone(nn.Module):
    """
    ResNet-18 up to layer4, returning a stride-32 feature map.
    Output: (N, 512, S, S), where S = img_size // 32.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        r = tvm.resnet18(
            weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.features = nn.Sequential(
            r.conv1,
            r.bn1,
            r.relu,
            r.maxpool,
            r.layer1,
            r.layer2,
            r.layer3,
            r.layer4,
        )

    def forward(self, x):
        return self.features(x)


class SharedConv(nn.Module):
    """
    Shared conv block placed after the backbone and before branching.
    """

    def __init__(self, in_ch=512, out_ch=512):
        super().__init__()
        self.layers = nn.Sequential(
            _cbr(in_ch, out_ch, k=1, p=0),
            _cbr(out_ch, out_ch, k=3, p=1),
            _cbr(out_ch, out_ch, k=3, p=1),
        )

    def forward(self, x):
        return self.layers(x)


class BranchHead(nn.Module):
    """
    One PLN branch: conv -> dilation stack -> inference head.
    """

    DILATION_RATES = (2, 2, 4, 8, 16, 1, 1)

    def __init__(self, in_ch, mid_ch, S, C, B=2):
        super().__init__()
        self.S = S
        self.C = C
        self.B = B
        self.slot_ch = 1 + C + 2 + S + S

        self.conv = _cbr(in_ch, mid_ch, k=3, p=1)

        dilated = []
        for rate in self.DILATION_RATES:
            dilated.append(_cbr(mid_ch, mid_ch, k=3, p=rate, d=rate))
        self.dilated = nn.Sequential(*dilated)

        self.out_conv = nn.Conv2d(mid_ch, 2 * B * self.slot_ch, kernel_size=3, padding=1)

    def forward(self, x):
        """
        x: (N, in_ch, S, S)
        """
        x = self.conv(x)
        x = self.dilated(x)
        raw = self.out_conv(x)

        N, _, Sh, Sw = raw.shape
        B, C, S = self.B, self.C, self.S
        assert Sh == S and Sw == S, f"Grid mismatch: expected {S}, got {Sh}x{Sw}"

        raw = raw.permute(0, 2, 3, 1)
        raw = raw.reshape(N, S, S, 2 * B, self.slot_ch)

        P_logit = raw[..., 0]
        Q_logit = raw[..., 1 : 1 + C]
        xy_logit = raw[..., 1 + C : 1 + C + 2]
        Lx_logit = raw[..., 1 + C + 2 : 1 + C + 2 + S]
        Ly_logit = raw[..., 1 + C + 2 + S :]

        return {
            "P":        torch.sigmoid(P_logit),
            "Q":        torch.softmax(Q_logit, dim=-1),
            "xy":       torch.sigmoid(xy_logit),
            "Lx":       torch.softmax(Lx_logit, dim=-1),
            "Ly":       torch.softmax(Ly_logit, dim=-1),
            "P_logit":  P_logit,
            "Q_logit":  Q_logit,
            "Lx_logit": Lx_logit,
            "Ly_logit": Ly_logit,
        }


_IMG_TO_S = {448: 14, 512: 16, 640: 20}
_BRANCH_NAMES = ("lt", "rt", "lb", "rb")
_SHARED_CH = 512   # ResNet-18 输出 512 通道，与 backbone 保持一致
_BRANCH_CH = 256   # 分支中间通道，适配 512 输入


class PLN(nn.Module):
    """
    PLN with ResNet-18 backbone and Figure 2 style branch layout.
    """

    def __init__(self, num_classes=20, img_size=448, B=2, pretrained=True):
        super().__init__()
        assert img_size in _IMG_TO_S, f"img_size must be one of {list(_IMG_TO_S)}"

        self.C = num_classes
        self.B = B
        self.img_size = img_size
        self.S = _IMG_TO_S[img_size]

        self.backbone = ResNet18Backbone(pretrained=pretrained)
        self.shared = SharedConv(in_ch=512, out_ch=_SHARED_CH)
        self.branches = nn.ModuleDict(
            {
                name: BranchHead(
                    in_ch=_SHARED_CH,
                    mid_ch=_BRANCH_CH,
                    S=self.S,
                    C=num_classes,
                    B=B,
                )
                for name in _BRANCH_NAMES
            }
        )

    def forward(self, x):
        """
        x: (N, 3, img_size, img_size)
        Returns:
            {"lt": {...}, "rt": {...}, "lb": {...}, "rb": {...}}
        """
        feat = self.backbone(x)
        feat = self.shared(feat)
        return {name: branch(feat) for name, branch in self.branches.items()}


if __name__ == "__main__":
    for img_size in [448, 512, 640]:
        S = _IMG_TO_S[img_size]
        m = PLN(num_classes=20, img_size=img_size, B=2, pretrained=False).eval()
        x = torch.zeros(1, 3, img_size, img_size)
        with torch.no_grad():
            out = m(x)
        for br, preds in out.items():
            assert preds["P"].shape == (1, S, S, 4), f"{br}: P shape wrong"
            assert preds["Q"].shape == (1, S, S, 4, 20), f"{br}: Q shape wrong"
            assert preds["xy"].shape == (1, S, S, 4, 2), f"{br}: xy shape wrong"
            assert preds["Lx"].shape == (1, S, S, 4, S), f"{br}: Lx shape wrong"
            assert preds["Ly"].shape == (1, S, S, 4, S), f"{br}: Ly shape wrong"
        params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"img={img_size} S={S} params={params:.1f}M shapes OK")
