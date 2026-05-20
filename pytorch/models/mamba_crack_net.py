"""MambaCrackNet: hybrid Mamba + convolutional U-Net for crack segmentation.

Faithful PyTorch port of the original TensorFlow `Mamba_unet` function. The model
keeps two parallel pathways:

* a token pathway processed by `MambaResidualBlock`s, with `PatchMerging` /
  `PatchExpanding` providing pyramidal down/up-sampling, and
* an image pathway built from `ConvResidualBlock`s that runs alongside the
  token stream, exchanging information at every scale through patchify/unpatchify
  bridges and pyramid fusion.

Inputs are NCHW tensors and outputs are per-pixel logits of shape (B, n_labels, H, W).
"""

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvResidualBlock
from .mamba import MambaResidualBlock
from .patches import PatchEmbedding, PatchExpanding, PatchExtract, PatchMerging


def _kernel_one_upsample(in_ch: int, out_ch: int, stride: int) -> nn.ConvTranspose2d:
    """ConvTranspose2d configured to keep TF 'same' semantics for kernel=1.

    For stride s>=1 with kernel=1 and padding=0, output_padding=s-1 yields
    output spatial dim = input * s.
    """
    output_padding = max(stride - 1, 0)
    return nn.ConvTranspose2d(
        in_ch,
        out_ch,
        kernel_size=1,
        stride=stride,
        padding=0,
        output_padding=output_padding,
    )


class MambaCrackNet(nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, int] = (512, 512),
        in_channels: int = 3,
        filter_num_begin: int = 64,
        depth: int = 4,
        patch_size: Tuple[int, int] = (4, 4),
        n_labels: int = 2,
        d_state: int = 16,
        expand: int = 2,
    ):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.depth = depth
        self.patch_size = patch_size
        self.n_labels = n_labels

        H, W = input_size
        ph, pw = patch_size
        assert H % ph == 0 and W % pw == 0, "input size must be divisible by patch size"

        # ---- encoder bookkeeping ------------------------------------------------
        embed_dim = filter_num_begin
        num_patch_h = H // ph
        num_patch_w = W // pw

        # initial token path
        self.patch_extract0 = PatchExtract(patch_size)
        self.patch_embed0 = PatchEmbedding(
            num_patches=num_patch_h * num_patch_w,
            patch_dim=in_channels * ph * pw,
            embed_dim=embed_dim,
        )
        self.token_block0 = MambaResidualBlock(
            embed_dim, expand=expand, d_conv=4, d_state=d_state
        )

        # initial image path
        self.image_block0 = ConvResidualBlock(in_channels, 32, downpool=False, uppool=False)

        # Track embed/grid as we descend so we can build matching modules
        self.encoder_patch_merge = nn.ModuleList()
        self.encoder_image_blocks = nn.ModuleList()
        self.encoder_image_patch_extract = nn.ModuleList()
        self.encoder_image_patch_embed = nn.ModuleList()
        self.encoder_token_blocks = nn.ModuleList()
        self.encoder_fusion_linear = nn.ModuleList()

        self.encoder_grid_sizes: List[Tuple[int, int]] = [(num_patch_h, num_patch_w)]
        self.encoder_embed_dims: List[int] = [embed_dim]
        self.encoder_image_channels: List[int] = [32]

        prev_img_ch = 32
        for i in range(depth - 1):
            # patch merging on tokens
            self.encoder_patch_merge.append(
                PatchMerging((num_patch_h, num_patch_w), embed_dim)
            )
            embed_dim *= 2
            num_patch_h //= 2
            num_patch_w //= 2

            # image residual block (downpool)
            out_img_ch = 64 * (2 * i + 1)
            self.encoder_image_blocks.append(
                ConvResidualBlock(prev_img_ch, out_img_ch, downpool=True, uppool=False)
            )

            # patchify the downsampled image to inject into the token stream
            self.encoder_image_patch_extract.append(PatchExtract(patch_size))
            self.encoder_image_patch_embed.append(
                PatchEmbedding(
                    num_patches=num_patch_h * num_patch_w,
                    patch_dim=out_img_ch * ph * pw,
                    embed_dim=embed_dim,
                )
            )

            # token block + fusion linear (concat of two embed_dim streams -> embed_dim)
            self.encoder_token_blocks.append(
                MambaResidualBlock(embed_dim, expand=expand, d_conv=2, d_state=d_state)
            )
            self.encoder_fusion_linear.append(
                nn.Linear(embed_dim * 2, embed_dim, bias=False)
            )

            self.encoder_grid_sizes.append((num_patch_h, num_patch_w))
            self.encoder_embed_dims.append(embed_dim)
            self.encoder_image_channels.append(out_img_ch)
            prev_img_ch = out_img_ch

        # ---- decoder bookkeeping ------------------------------------------------
        # Reverse the per-scale lists once so iteration follows the upsampling order.
        rev_grid = self.encoder_grid_sizes[::-1]  # [(16,16),(32,32),(64,64),(128,128)]
        rev_embed = self.encoder_embed_dims[::-1]  # [512,256,128,64]
        rev_img_ch = self.encoder_image_channels[::-1]  # [320,192,64,32]

        depth_decode = depth - 1
        self.decoder_patch_expand = nn.ModuleList()
        self.decoder_image_uppool_blocks = nn.ModuleList()
        self.decoder_concat_linear = nn.ModuleList()
        self.decoder_token_blocks = nn.ModuleList()
        self.decoder_token_proj = nn.ModuleList()
        self.decoder_pyramid_expand = nn.ModuleList()
        self.decoder_image_pyramid_blocks = nn.ModuleList()

        # state going into the decoder (from the deepest encoder scale)
        cur_img_ch = rev_img_ch[0]
        for i in range(depth_decode):
            cur_embed = rev_embed[i]  # incoming token channel
            next_embed = rev_embed[i + 1]
            cur_grid = rev_grid[i]
            next_grid = rev_grid[i + 1]

            self.decoder_patch_expand.append(
                PatchExpanding(cur_grid, cur_embed, upsample_rate=2, return_vector=True)
            )

            # image up-pool block: concat current Img_X with skip image, output = next_embed
            skip_img_ch = rev_img_ch[i]
            self.decoder_image_uppool_blocks.append(
                ConvResidualBlock(
                    cur_img_ch + skip_img_ch,
                    next_embed,
                    downpool=False,
                    uppool=True,
                )
            )
            cur_img_ch = next_embed  # after uppool block

            # concat tokens with the matching encoder skip then project back to next_embed
            self.decoder_concat_linear.append(
                nn.Linear(next_embed * 2, next_embed, bias=False)
            )

            self.decoder_token_blocks.append(
                MambaResidualBlock(next_embed, expand=expand, d_conv=2, d_state=d_state)
            )
            self.decoder_token_proj.append(nn.Linear(next_embed, next_embed, bias=False))

            # token -> image pyramid (unpatchify with patch_size as upsample factor)
            self.decoder_pyramid_expand.append(
                PatchExpanding(
                    next_grid,
                    next_embed,
                    upsample_rate=ph,
                    return_vector=False,
                )
            )
            pyramid_img_ch = next_embed // ph

            # fuse uppooled image with the pyramid image
            pyramid_out_ch = 64 * (2 * (depth_decode - i))
            self.decoder_image_pyramid_blocks.append(
                ConvResidualBlock(
                    cur_img_ch + pyramid_img_ch,
                    pyramid_out_ch,
                    downpool=False,
                    uppool=False,
                )
            )
            cur_img_ch = pyramid_out_ch

        # ---- pyramid head -------------------------------------------------------
        pyramid_channels = [64 * (2 * (depth_decode - i)) for i in range(depth_decode)]
        self.pyramid_conv = nn.ModuleList(
            [nn.Conv2d(c, 64, kernel_size=3, padding=1) for c in pyramid_channels]
        )
        self.pyramid_upsample = nn.ModuleList(
            [
                _kernel_one_upsample(64, 32, stride=2 ** (depth_decode - 1 - i))
                for i in range(depth_decode)
            ]
        )

        self.head_conv1 = nn.Conv2d(32 * depth_decode, 16, kernel_size=2, padding="same")
        self.head_conv2 = nn.Conv2d(16, n_labels, kernel_size=1)

    # -------------------------------------------------------------------------
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, in_channels, H, W) -> logits (B, n_labels, H, W)."""
        # ---- initial scale ---------------------------------------------------
        tokens = self.patch_extract0(img)
        tokens = self.patch_embed0(tokens)
        tokens = self.token_block0(tokens)

        img_feat = self.image_block0(img)

        token_skips = [tokens]
        image_skips = [img_feat]

        # ---- encoder loop ----------------------------------------------------
        for i in range(self.depth - 1):
            tokens = self.encoder_patch_merge[i](tokens)

            img_feat = self.encoder_image_blocks[i](img_feat)
            image_skips.append(img_feat)

            img_tokens = self.encoder_image_patch_extract[i](img_feat)
            img_tokens = self.encoder_image_patch_embed[i](img_tokens)

            t = self.encoder_token_blocks[i](tokens)
            tokens = torch.cat([t, img_tokens], dim=-1)
            tokens = self.encoder_fusion_linear[i](tokens)

            token_skips.append(tokens)

        # ---- decoder loop ----------------------------------------------------
        token_skips = token_skips[::-1]
        image_skips = image_skips[::-1]

        tokens = token_skips[0]
        decode_skips = token_skips[1:]
        pyramid_feats: List[torch.Tensor] = []

        for i, (expand, up_block, concat_lin, tok_block, tok_proj, pyr_expand, pyr_block) in enumerate(
            zip(
                self.decoder_patch_expand,
                self.decoder_image_uppool_blocks,
                self.decoder_concat_linear,
                self.decoder_token_blocks,
                self.decoder_token_proj,
                self.decoder_pyramid_expand,
                self.decoder_image_pyramid_blocks,
            )
        ):
            tokens = expand(tokens)

            img_feat = up_block(torch.cat([img_feat, image_skips[i]], dim=1))

            tokens = torch.cat([tokens, decode_skips[i]], dim=-1)
            tokens = concat_lin(tokens)
            tokens = tok_block(tokens)
            tokens = tok_proj(tokens)

            pyramid_img = pyr_expand(tokens)  # NCHW
            img_feat = pyr_block(torch.cat([img_feat, pyramid_img], dim=1))
            pyramid_feats.append(img_feat)

        # ---- pyramid head ----------------------------------------------------
        processed = []
        for i, feat in enumerate(pyramid_feats):
            x = F.relu(self.pyramid_conv[i](feat))
            x = F.relu(self.pyramid_upsample[i](x))
            processed.append(x)

        fused = torch.cat(processed, dim=1)
        fused = F.relu(self.head_conv1(fused))
        logits = self.head_conv2(fused)
        return logits
