import torch
import torch.nn as nn
import torch.nn.functional as F


def _stable_attention(query, key):
    """Compute channel-wise spatial attention without FP16 overflow."""
    # The original Q @ K reduction spans a spatial axis. On 224px patches its
    # logits can exceed FP16's finite range after only a few optimizer steps.
    # Disable autocast locally and use the standard dot-product scale.
    with torch.autocast(device_type=query.device.type, enabled=False):
        query_fp32 = torch.nan_to_num(
            query.float(), nan=0.0, posinf=1e4, neginf=-1e4
        ).clamp_(-1e4, 1e4).permute(0, 1, 3, 2)
        key_fp32 = torch.nan_to_num(
            key.float(), nan=0.0, posinf=1e4, neginf=-1e4
        ).clamp_(-1e4, 1e4)
        scale = key_fp32.shape[-2] ** -0.5
        return torch.softmax(torch.matmul(query_fp32, key_fp32) * scale, dim=-1)

class MCAM(nn.Module):
    def __init__(self, in_channels):
        super(MCAM, self).__init__()

        self.in_channels = in_channels
        # self.inter_channels = inter_channels if inter_channels else in_channels // 2

        # 1x1卷积层，用于从输入特征图中提取VOPT、QOPT和KOPT
        self.VOPT_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.QOPT_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.KOPT_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 1x1卷积层，用于从输入特征图中提取VSAR、QSAR和KSAR
        self.VSAR_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.QSAR_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.KSAR_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, sar_features, optical_features):
        batch_size = optical_features.size(0)

        # 提取光学图像特征的VOPT、QOPT和KOPT
        VOPT = self.VOPT_conv(optical_features)
        QOPT = self.QOPT_conv(optical_features)
        KOPT = self.KOPT_conv(optical_features)

        # 提取SAR图像特征的VSAR、QSAR和KSAR
        VSAR = self.VSAR_conv(sar_features)
        QSAR = self.QSAR_conv(sar_features)
        KSAR = self.KSAR_conv(sar_features)

        # 计算光学图像特征的自注意力图Shigh_OPT
        S_OPT = _stable_attention(QOPT, KOPT)

        # 计算SAR图像特征的自注意力图Shigh_SAR
        S_SAR = _stable_attention(QSAR, KSAR)

        # 计算光学和SAR图像特征的交叉融合注意力图Shigh_cro
        with torch.autocast(device_type=VOPT.device.type, enabled=False):
            S_cro = S_OPT * S_SAR

            # Keep the multiplicative fusion in FP32 too: two feature maps are
            # multiplied here, which has the same overflow risk as Q @ K.
            value_opt = torch.nan_to_num(
                VOPT.float(), nan=0.0, posinf=1e4, neginf=-1e4
            ).clamp_(-1e4, 1e4)
            value_sar = torch.nan_to_num(
                VSAR.float(), nan=0.0, posinf=1e4, neginf=-1e4
            ).clamp_(-1e4, 1e4)
            Att_OPT = S_cro * value_opt
            Att_SAR = S_cro * value_sar
            Att_OPT_SAR = torch.nan_to_num(
                Att_OPT * Att_SAR,
                nan=0.0,
                posinf=1e4,
                neginf=-1e4,
            ).clamp_(-1e4, 1e4)

        return Att_OPT_SAR


if __name__ == '__main__':
    model = MCAM(in_channels=256)
    model.train()
    sar = torch.randn(2, 256, 64, 64)
    opt = torch.randn(2, 256, 64, 64)
    print(model)
    print("input:", sar.shape, opt.shape)
    print("output:", model(sar, opt).shape)
