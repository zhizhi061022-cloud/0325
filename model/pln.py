"""
PLN model – supports ResNet-18 and InceptionV2 (BN-Inception) backbones.

Architecture (Figure 2 of the PLN paper):
    backbone -> shared conv stack -> 4 branches
    each branch: conv -> dilation stack -> inference head

Paper-faithful channel sizes (Fig.2, img_size=448):
    Backbone output : 14×14×3328  (TF inception_v2 multi-block fusion)
    SharedConv out  : 14×14×1536  (K: 1×1→3×3→3×3, all 1536ch)
    BranchHead out  : 14×14×204   (2·B·slot_ch = 2·2·51 = 204)

Our timm implementation:
    inception_v4: stride-32, 1536ch output — directly matches SharedConv input.
    (bninception was removed from modern timm; inception_v4 is a stronger
     backbone and its 1536ch output is architecturally equivalent)

Backbone choices:
    'resnet18'   – torchvision ResNet-18, stride-32, 512ch out
    'inceptionv4'– timm InceptionV4, stride-32, 1536ch out
                   (requires: pip install timm)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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

    out_channels = 512

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


class InceptionV4Backbone(nn.Module):
    """
    InceptionV4 backbone via timm.
    Output: (N, 1536, S, S), stride-32, where S = img_size // 32.

    1536ch directly matches the PLN paper's SharedConv input (Fig.2).
    InceptionV4 is available in modern timm and is a stronger backbone
    than the original BN-Inception used in the paper.

    Install dependency: pip install timm
    """

    out_channels = 1536

    def __init__(self, pretrained=True):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError(
                "InceptionV4 backbone requires timm: pip install timm"
            )
        # forward_features() returns spatial map before global avg-pool
        self._m = timm.create_model('inception_v4', pretrained=pretrained)

    def forward(self, x):
        return self._m.forward_features(x)  # (N, 1536, S, S)


class SharedConv(nn.Module):
    """
    Shared conv block placed after the backbone and before branching.
    The 1×1 conv at the front projects backbone channels to out_ch.
    """

    def __init__(self, in_ch, out_ch=512):
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
# Paper Fig.2: SharedConv outputs 1536ch; BranchHead uses 1536ch mid_ch.
# Previously set to 512/256 (severely under-parameterized vs. paper).
_SHARED_OUT_CH = 1536  # matches paper SharedConv output channels
_BRANCH_CH     = 1536  # matches paper BranchHead intermediate channels

_BACKBONES = {
    'resnet18':    ResNet18Backbone,
    'inceptionv4': InceptionV4Backbone,
}


class PLN(nn.Module):
    """
    PLN with selectable backbone (resnet18 | inceptionv2).
    Figure 2 layout: backbone -> shared conv -> 4 branches.
    """

    def __init__(self, num_classes=20, img_size=448, B=2,
                 pretrained=True, backbone='resnet18'):
        super().__init__()
        assert img_size in _IMG_TO_S, f"img_size must be one of {list(_IMG_TO_S)}"
        assert backbone in _BACKBONES, \
            f"backbone must be one of {list(_BACKBONES)}, got '{backbone}'"

        self.C = num_classes
        self.B = B
        self.img_size = img_size
        self.S = _IMG_TO_S[img_size]

        self.backbone = _BACKBONES[backbone](pretrained=pretrained)
        bb_ch = self.backbone.out_channels   # 512 for resnet18, 1024 for inceptionv2

        self.shared = SharedConv(in_ch=bb_ch, out_ch=_SHARED_OUT_CH)
        self.branches = nn.ModuleDict(
            {
                name: BranchHead(
                    in_ch=_SHARED_OUT_CH,
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
    for bb in ['resnet18', 'inceptionv4']:
        for img_size in [448, 512, 640]:
            S = _IMG_TO_S[img_size]
            try:
                m = PLN(num_classes=20, img_size=img_size, B=2,
                        pretrained=False, backbone=bb).eval()
            except ImportError as e:
                print(f"[{bb}] img={img_size}: SKIP – {e}")
                continue
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
            print(f"[{bb}] img={img_size} S={S} params={params:.1f}M shapes OK")
