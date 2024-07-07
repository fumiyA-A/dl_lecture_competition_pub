import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )
        
        self.blocks.apply(self.init_weights)

        self.head = nn.Sequential(
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )
        self.head.apply(self.init_weights)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)
    
    def init_weights(self, m):  # add He initialization 7/7
        if type(m) == nn.Linear or type(m) == nn.Conv1d:
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.0)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.2, # change 0.1->0.2
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same") 
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same") 
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.adavgpool = nn.AdaptiveAvgPool1d(1) # add 7/7
        self.dropout = nn.Dropout(p_drop)
        

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))
        
        X = self.adavgpool(X) # add 7/7
        X = self.dropout(X) # add dropout 7/7

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        X = self.adavgpool(X) # add 7/7
        return self.dropout(X)