import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group
from RST import _C



class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s) 
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

class RST(nn.Module):
    def __init__(self, args, output_channels=40):
        super(RST, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Rough_Set_Transformer(args)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)

        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x

class Rough_Set_Transformer(nn.Module):
    def __init__(self, args, channels=256):
        super(Rough_Set_Transformer, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.ra1 = RoughsetAttention(channels)
        self.ra2 = RoughsetAttention(channels)
        self.ra3 = RoughsetAttention(channels)
        self.ra4 = RoughsetAttention(channels)

    def forward(self, x):

        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.ra1(x)
        x2 = self.ra2(x1)
        x3 = self.ra3(x2)
        x4 = self.ra4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

def Granulation(q, k):
    matA_square = q ** 2. @ torch.ones(k.shape[-2:]).cuda()
    matB_square = torch.ones(q.shape[-2:]).cuda() @ k ** 2.
    return matA_square + matB_square - 2. * (q @ k)

class RoughsetAttention(nn.Module):
    def __init__(self, channels):
        super(RoughsetAttention, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)

        batch_size,  q_len, input_channels = x_q.size()
        batch_size,  input_channels, k_len = x_k.size()
        output = x_q.new(batch_size, 1, q_len, k_len)

        kernel = Granulation(x_q, x_k)
        kernel = torch.exp(-kernel / 2)
        kernel = kernel.reshape(batch_size, 1, q_len, k_len)

        _C.Approximation(kernel, kernel, output)
        output = output.reshape(batch_size, q_len, k_len)



        x_r = torch.bmm(output,x_v)


        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
