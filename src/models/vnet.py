# import torch
# from torch import nn
# from torch.utils.checkpoint import checkpoint_sequential
# from src.models.layers import InputTransition, DownTransition, UpTransition, OutputTransition

# class VNet(nn.Module):
#     # the number of convolutions in each layer corresponds
#     # to what is in the actual prototxt, not the intent
#     def __init__(self=True):
#         super(VNet, self).__init__()
#         self.in_tr = InputTransition(16)
#         self.down_tr32 = DownTransition(16, 1)
#         self.down_tr64 = DownTransition(32, 2)
#         self.down_tr128 = DownTransition(64, 3, dropout=True)
#         self.down_tr256 = DownTransition(128, 2, dropout=True)
#         self.up_tr256 = UpTransition(256, 256, 2, dropout=True)
#         self.up_tr128 = UpTransition(256, 128, 2, dropout=True)
#         self.up_tr64 = UpTransition(128, 64, 1)
#         self.up_tr32 = UpTransition(64, 32, 1)
#         self.out_tr = OutputTransition(32, nll)

#     def forward(self, x):
#         out16 = self.in_tr(x)
#         out32 = self.down_tr32(out16)
#         out64 = self.down_tr64(out32)
#         out128 = self.down_tr128(out64)
#         out256 = self.down_tr256(out128)
#         out = self.up_tr256(out256, out128)
#         out = self.up_tr128(out, out64)
#         out = self.up_tr64(out, out32)
#         out = self.up_tr32(out, out16)
#         out = self.out_tr(out)
#         return out