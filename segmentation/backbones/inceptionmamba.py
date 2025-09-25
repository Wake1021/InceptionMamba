import torch
import torch.nn as nn

import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from fvcore.nn import FlopCountAnalysis
from timm.models.helpers import checkpoint_seq
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple

from backbones.lib_mamba.vmambanew import SS2D

from mmseg.registry import MODELS

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1e-6, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class BottleneckMamba(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.base_scale = _ScaleModule([1, c_, 1, 1])
        self.ss2d = SS2D(
            d_model=c_,
            d_state=1,
            ssm_ratio=1,
            initialize="v2",
            forward_type="v05",  # 交叉扫描优化版本
            channel_first=True,  # 匹配卷积的输入格式 [B,C,H,W]
        )
        self.cv2 = nn.Conv2d(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.base_scale(self.ss2d(self.cv1(x)))) if self.add else self.cv2(
            self.base_scale(self.ss2d(self.cv1(x))))


class SEModule(nn.Module):
    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            inner_act(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            out_act(),
        )

    def forward(self, x):
        x = x * self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, c, use_bnck="SE", e=0.5):
        super().__init__()
        assert (c % 8 == 0)
        self.c = c // 8

        self.conv3x3 = nn.Conv2d(self.c, self.c, 3, 1, 1, groups=self.c)
        self.conv11x3 = nn.Conv2d(self.c, self.c, (11, 3), 1, (5, 1), groups=self.c)
        self.conv3x11 = nn.Conv2d(self.c, self.c, (3, 11), 1, (1, 5), groups=self.c)
        self.split_channels = [self.c, self.c, 6 * self.c]
        self.use_bnck = use_bnck

        if use_bnck == "BMamba":
            self.block = BottleneckMamba(c, c, e=0.5)
        elif use_bnck == "SE":
            self.block = SEModule(c)

    def forward(self, x):
        x1, x2, x3 = torch.split(x, self.split_channels, dim=1)

        x1 = self.conv3x3(x1)
        x2_1 = self.conv11x3(x2)
        x2_2 = self.conv3x11(x2)
        x2 = x2_1 + x2_2

        x = torch.cat((x1, x2, x3), dim=1)

        if self.use_bnck is not None:
            x = self.block(x)
        return x


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """

    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
            norm_layer=None, bias=True, drop=0., use_dwconv3x3=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])
        self.use_dwconv3x3 = use_dwconv3x3
        if use_dwconv3x3:
            self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)

        if self.use_dwconv3x3:
            x = self.dwconv(x) + x

        x = self.drop(x)
        x = self.fc2(x)
        return x

class MlpHead(nn.Module):
    """ MLP classification head
    """

    def __init__(self, dim, num_classes=1000, mlp_ratio=3, act_layer=nn.GELU, drop=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.norm = nn.BatchNorm2d(dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = x.mean((2, 3))  # global average pooling
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x



class InceptionMambaBlock(nn.Module):
    """InceptionMambaBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            token_mixer=nn.Identity,
            mlp_layer=ConvMlp,
            mlp_ratio=4,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,
            use_bnck = "SE",
            use_dwconv3x3=False,
    ):
        super().__init__()
        self.token_mixer = token_mixer(dim, use_bnck)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer, use_dwconv3x3=use_dwconv3x3)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x


class InceptionMambaStage(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            ds_stride=2,
            depth=2,
            drop_path_rates=None,
            ls_init_value=1.0,
            token_mixer=nn.Identity,
            act_layer=nn.GELU,
            mlp_ratio=4,
            use_bnck = "SE",
            use_dwconv3x3 = False,
    ):
        super().__init__()
        self.grad_checkpointing = False
        if ds_stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_chs),
            )
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(InceptionMambaBlock(
                dim=out_chs,
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                token_mixer=token_mixer,
                act_layer=act_layer,
                mlp_ratio=mlp_ratio,
                use_bnck=use_bnck,
                use_dwconv3x3=use_dwconv3x3,
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


def stem(in_chans=3, embed_dim=96):
    return nn.Sequential(
        nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim // 2),
        nn.GELU(),
        nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim // 2),
        nn.GELU(),
        nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim),
        nn.GELU(),
        nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim)
    )

@MODELS.register_module()
class InceptionMamba(nn.Module):
    r"""
    Args:
        arch (str): InceptionMamba architecture choosing from 'tiny', 'small','base' and 'large'. Defaults to 'tiny'
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        norm_layer: Normalziation layer. Default: nn.BatchNorm2d
        act_layer: Activation function for MLP. Default: nn.GELU
        mlp_ratios (int or tuple(int)): MLP ratios. Default: (4, 4, 4, 3)
        head_fn: classifier head
        drop_rate (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                    {'depths': [3, 3, 12, 3],
                         'dims': [72, 144, 288, 576],
                         'use_bnck': ["BMamba", "BMamba", "BMamba", "BMamba"],
                         # 'use_bnck': [None, None, None, None],
                         'use_dwconv3x3': [False, False, False, False],
                         'token_mixers': Block,
                         }),
        **dict.fromkeys(['s', 'small'],
                        {'depths':[4, 4, 32, 4],
                        'dims': [72, 144, 288, 576],
                        'use_bnck': ["BMamba", "BMamba", "BMamba", "BMamba"],
                        'use_dwconv3x3': [False, False, False, False],
                         'token_mixers': Block,
                         }),
        **dict.fromkeys(['b', 'base'],
                        {'depths': [4, 4, 34, 4],
                         'dims': [96, 192, 384, 768],
                         'use_bnck': ["BMamba", "BMamba", "BMamba", "BMamba"],
                         'use_dwconv3x3': [False, False, False, False],
                         'token_mixers': Block,
                         }),
    }
    def __init__(
            self,
            arch='tiny',
            in_chans=3,
            num_classes=1000,
            act_layer=nn.GELU,
            mlp_ratios=(4, 4, 4, 4),
            drop_rate=0.,
            drop_path_rate=0.,
            ls_init_value=1e-6,
            pretrained=None,
            **kwargs,
    ):
        super().__init__()

        self.pretrained = pretrained

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {'depths', 'dims', 'use_bnck', 'use_dwconv3x3', 'token_mixers'}
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.depths = self.arch_settings['depths']
        self.dims = self.arch_settings['dims']
        self.use_bnck = self.arch_settings['use_bnck']
        self.use_dwconv3x3 = self.arch_settings['use_dwconv3x3']
        self.token_mixers = self.arch_settings['token_mixers']

        num_stage = len(self.depths)

        if not isinstance(self.token_mixers, (list, tuple)):
            token_mixers = [self.token_mixers] * num_stage
        if not isinstance(mlp_ratios, (list, tuple)):
            mlp_ratios = [mlp_ratios] * num_stage

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.stem = stem(in_chans=in_chans, embed_dim=self.dims[0])

        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(self.depths)).split(self.depths)]
        stages = []
        prev_chs = self.dims[0]
        # feature resolution stages, each consisting of multiple residual blocks
        for i in range(num_stage):
            out_chs = self.dims[i]
            stages.append(InceptionMambaStage(
                prev_chs,
                out_chs,
                ds_stride=2 if i > 0 else 1,
                depth=self.depths[i],
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                token_mixer=token_mixers[i],
                mlp_ratio=mlp_ratios[i],
                use_bnck=self.use_bnck[i],
                use_dwconv3x3=self.use_dwconv3x3[i],
            ))
            prev_chs = out_chs
        self.stages = nn.Sequential(*stages)
        self.num_features = prev_chs
        self.apply(self._init_weights)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def forward(self, x):
        x = self.stem(x)

        out = []
        for block in self.stages:
            x = block(x)
            out.append(x)

        return out

    def _init_weights(self, m):
        if self.pretrained is None:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        else:
            state_dict = torch.load(self.pretrained, map_location='cpu')
            self_state_dict = self.state_dict()
            for k, v in state_dict.items():
                if k in self_state_dict.keys():
                    self_state_dict.update({k: v})
            self.load_state_dict(self_state_dict, strict=True)