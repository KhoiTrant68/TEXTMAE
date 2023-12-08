import warnings

warnings.filterwarnings("ignore")

from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, ConvTranspose2d
from pytorch_msssim import SSIM

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import conv3x3, subpel_conv3x3
from compressai.models import CompressionModel
from compressai.ops import quantize_ste

from common.pos_embed import get_2d_sincos_pos_embed
from loss.vgg import normalize_batch, de_normalize, feature_network

from timm.models.vision_transformer import PatchEmbed, Block


class MAEC(CompressionModel):
    """
    Masked Autoencoder with Vision Transformer backbone

    This class inherits from MaskedAutoencoder in *facebookresearch/mae* class.
    See the original paper and the `MAE' documentation
    <https://github.com/facebookresearch/mae/blob/main/README.md> for an introduction.
    """

    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=3,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_pix_loss=False,
        latent_depth=384,
        hyperprior_depth=192,
        num_slices=12,
        num_keep_patches=144,
    ):
        super().__init__()

        # Initialize frozen stage
        self.frozen_stages = -1

        # Model hyperparameters
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_depth = encoder_depth
        self.encoder_num_heads = encoder_num_heads
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.latent_depth = latent_depth
        self.hyperprior_depth = hyperprior_depth
        self.num_slices = num_slices
        self.num_keep_patches = num_keep_patches

        # Entropy model
        self.entropy_bottleneck = EntropyBottleneck(hyperprior_depth)
        self.gaussian_conditional = GaussianConditional(None)
        self.max_support_slices = self.num_slices // 2

        ## Compression Modules
        # G_a Module
        self.g_a = nn.Sequential(
            nn.Conv2d(
                self.encoder_embed_dim,
                int(
                    self.decoder_embed_dim
                    + (self.encoder_embed_dim - self.decoder_embed_dim) * 3 / 4
                ),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GELU(),
            nn.Conv2d(
                int(
                    self.decoder_embed_dim
                    + (self.encoder_embed_dim - self.decoder_embed_dim) * 3 / 4
                ),
                int(
                    self.decoder_embed_dim
                    + (self.encoder_embed_dim - self.decoder_embed_dim) * 2 / 4
                ),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GELU(),
            nn.Conv2d(
                int(
                    self.decoder_embed_dim
                    + (self.encoder_embed_dim - self.decoder_embed_dim) * 2 / 4
                ),
                self.decoder_embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GELU(),
            nn.Conv2d(
                self.decoder_embed_dim,
                self.latent_depth,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GELU(),
        )

        # G_s Module
        self.g_s = nn.Sequential(
            nn.ConvTranspose2d(
                self.latent_depth,
                self.decoder_embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.decoder_embed_dim,
                int(
                    self.decoder_embed_dim
                    + (self.encoder_embed_dim - self.decoder_embed_dim) * 2 / 4
                ),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                int(
                    self.decoder_embed_dim
                    + (self.encoder_embed_dim - self.decoder_embed_dim) * 2 / 4
                ),
                int(
                    self.decoder_embed_dim
                    + (self.encoder_embed_dim - self.decoder_embed_dim) * 3 / 4
                ),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                int(
                    self.decoder_embed_dim
                    + (self.encoder_embed_dim - self.decoder_embed_dim) * 3 / 4
                ),
                self.encoder_embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GELU(),
        )

        # H_a Module
        self.h_a = nn.Sequential(
            conv3x3(self.latent_depth, self.latent_depth, stride=1),
            nn.GELU(),
            conv3x3(
                self.latent_depth,
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) * 3 / 4
                ),
                stride=1,
            ),
            nn.GELU(),
            conv3x3(
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) * 3 / 4
                ),
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) * 2 / 4
                ),
                stride=2,
            ),
            nn.GELU(),
            conv3x3(
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) * 2 / 4
                ),
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) / 4
                ),
                stride=1,
            ),
            nn.GELU(),
            conv3x3(
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) / 4
                ),
                self.hyperprior_depth,
                stride=2,
            ),
        )

        # H_s Module
        self.h_s_mean = nn.Sequential(
            conv3x3(
                self.hyperprior_depth,
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) / 4
                ),
                stride=1,
            ),
            nn.GELU(),
            subpel_conv3x3(
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) / 4
                ),
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) * 2 / 4
                ),
                r=2,
            ),
            nn.GELU(),
            conv3x3(
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) * 2 / 4
                ),
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) * 3 / 4
                ),
                stride=1,
            ),
            nn.GELU(),
            subpel_conv3x3(
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) * 3 / 4
                ),
                self.latent_depth,
                r=2,
            ),
            nn.GELU(),
            conv3x3(self.latent_depth, self.latent_depth, stride=1),
        )

        self.h_s_scale = nn.Sequential(
            conv3x3(
                self.hyperprior_depth,
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) / 4
                ),
                stride=1,
            ),
            nn.GELU(),
            subpel_conv3x3(
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) / 4
                ),
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) * 2 / 4
                ),
                r=2,
            ),
            nn.GELU(),
            conv3x3(
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) * 2 / 4
                ),
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) * 3 / 4
                ),
                stride=1,
            ),
            nn.GELU(),
            subpel_conv3x3(
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) * 3 / 4
                ),
                self.latent_depth,
                r=2,
            ),
            nn.GELU(),
            conv3x3(self.latent_depth, self.latent_depth, stride=1),
        )

        # CC_Transform Module
        self.cc_transform_mean = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        int(
                            self.latent_depth
                            + (self.latent_depth // self.num_slices)
                            * min(i, self.num_slices // 2)
                        ),
                        int(
                            self.latent_depth
                            // self.num_slices
                            * (self.num_slices // 2 + 1)
                        ),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        int(
                            self.latent_depth
                            // self.num_slices
                            * (self.num_slices // 2 + 1)
                        ),
                        int(
                            self.latent_depth
                            // self.num_slices
                            * (self.num_slices // 2 * 3 / 4 + 1)
                        ),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        int(
                            self.latent_depth
                            // self.num_slices
                            * (self.num_slices // 2 * 3 / 4 + 1)
                        ),
                        int(
                            self.latent_depth
                            // self.num_slices
                            * (self.num_slices // 2 * 2 / 4 + 1)
                        ),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        int(
                            self.latent_depth
                            // self.num_slices
                            * (self.num_slices // 2 * 2 / 4 + 1)
                        ),
                        int(
                            self.latent_depth
                            // self.num_slices
                            * (self.num_slices // 2 * 1 / 4 + 1)
                        ),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        int(
                            self.latent_depth
                            // self.num_slices
                            * (self.num_slices // 2 * 1 / 4 + 1)
                        ),
                        int(self.latent_depth // self.num_slices),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(self.num_slices)
            ]
        )

        self.cc_transform_scale = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        int(
                            self.latent_depth
                            + (self.latent_depth // self.num_slices)
                            * min(i, self.num_slices // 2)
                        ),
                        int(
                            self.latent_depth
                            // self.num_slices
                            * (self.num_slices // 2 + 1)
                        ),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        int(
                            self.latent_depth
                            // self.num_slices
                            * (self.num_slices // 2 + 1)
                        ),
                        int(
                            self.latent_depth
                            // self.num_slices
                            * (self.num_slices // 2 * 3 / 4 + 1)
                        ),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        int(
                            self.latent_depth
                            // self.num_slices
                            * (self.num_slices // 2 * 3 / 4 + 1)
                        ),
                        int(
                            self.latent_depth
                            // self.num_slices
                            * (self.num_slices // 2 * 2 / 4 + 1)
                        ),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        int(
                            self.latent_depth
                            // self.num_slices
                            * (self.num_slices // 2 * 2 / 4 + 1)
                        ),
                        int(
                            self.latent_depth
                            // self.num_slices
                            * (self.num_slices // 2 * 1 / 4 + 1)
                        ),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        int(
                            self.latent_depth
                            // self.num_slices
                            * (self.num_slices // 2 * 1 / 4 + 1)
                        ),
                        int(self.latent_depth // self.num_slices),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(self.num_slices)
            ]
        )

        # LRP Transform Module
        self.lrp_transform = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        int(
                            self.latent_depth
                            + (self.latent_depth // self.num_slices)
                            * min(i + 1, self.num_slices // 2 + 1)
                        ),
                        int(
                            self.latent_depth
                            // self.num_slices
                            * (self.num_slices // 2 + 1)
                        ),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        int(
                            self.latent_depth
                            // self.num_slices
                            * (self.num_slices // 2 + 1)
                        ),
                        int(
                            self.latent_depth
                            // self.num_slices
                            * (self.num_slices // 2 * 3 / 4 + 1)
                        ),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        int(
                            self.latent_depth
                            // self.num_slices
                            * (self.num_slices // 2 * 3 / 4 + 1)
                        ),
                        int(
                            self.latent_depth
                            // self.num_slices
                            * (self.num_slices // 2 * 2 / 4 + 1)
                        ),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        int(
                            self.latent_depth
                            // self.num_slices
                            * (self.num_slices // 2 * 2 / 4 + 1)
                        ),
                        int(
                            self.latent_depth
                            // self.num_slices
                            * (self.num_slices // 2 * 1 / 4 + 1)
                        ),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        int(
                            self.latent_depth
                            // self.num_slices
                            * (self.num_slices // 2 * 1 / 4 + 1)
                        ),
                        int(self.latent_depth // self.num_slices),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(self.num_slices)
            ]
        )

        # Initialize freeze stage status
        self._freeze_stages()

        # ------------------------------------ Initialize MAE layers ------------------------------------
        # ------------------ Encoder ------------------
        self.encoder_embed = PatchEmbed(
            img_size, patch_size, in_chans, self.encoder_embed_dim
        )
        num_patches = self.encoder_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.encoder_embed_dim), requires_grad=False
        )  # Fixed sin-cos embedding

        # Define encoder blocks
        self.encoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=self.encoder_embed_dim,
                    num_heads=self.encoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(self.encoder_depth)
            ]
        )
        self.encoder_norm = norm_layer(self.encoder_embed_dim)

        # ------------------ Decoder ------------------
        self.decoder_embed = nn.Linear(
            self.encoder_embed_dim, self.decoder_embed_dim, bias=True
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.decoder_embed_dim), requires_grad=False
        )  # Fixed sin-cos embedding

        # Define decoder blocks
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=self.decoder_embed_dim,
                    num_heads=self.decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(self.decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(self.decoder_embed_dim)

        self.decoder_pred = nn.Linear(
            self.decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )
        # ----------------------------------------------

        # Initialize norm_pix_loss
        self.norm_pix_loss = norm_pix_loss

        # Initialize weight
        self.initialize_weights()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.encoder_embed.eval()
            for param in self.encoder_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, vis_num, state_dict):
        net = cls(visual_tokens=vis_num)
        net.load_state_dict(state_dict)
        return net

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.encoder_pos_embed.shape[-1],
            int(self.encoder_embed.num_patches**0.5),
            cls_token=True,
        )
        self.encoder_pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.encoder_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.encoder_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        Patchify function that converts an image into a patched feature.

        Args:
            imgs (torch.Tensor): Batch of images with shape (N, 3, H, W).

        Returns:
            patched_feature (torch.Tensor): Patched feature with shape (N, L, D).
                - N: Number of batches
                - L: Number of patches, calculated as (H // patch_size) ** 2
                - D: Dimension of each patch, calculated as patch_size ** 2 * 3
        """
        patch_size = self.encoder_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0

        # Split the images into patches
        h = w = imgs.shape[2] // patch_size
        x = imgs.reshape(imgs.shape[0], 3, h, patch_size, w, patch_size)

        # Rearrange the patches
        x = torch.einsum("nchpwq->nhwpqc", x)

        # Reshape the patches into the final format
        patched_feature = x.reshape(imgs.shape[0], h * w, patch_size**2 * 3)
        return patched_feature

    def unpatchify(self, patched_feature):
        """
        Unpatchify function to convert from a patched feature to an image (N, 3, H, W).
        This function is typically used for results after passing through the decoder MAE.

        Args:
            patched_feature (torch.Tensor): Patched feature after decoding with shape (N, L, patch_size**2 * 3).

        Returns:
            imgs (torch.Tensor): Image with shape (N, 3, H, W).
        """
        patch_size = self.encoder_embed.patch_size[0]

        # Calculate the dimensions of the output image
        h = w = int(patched_feature.shape[1] ** 0.5)
        assert h * w == patched_feature.shape[1]

        # Reshape and rearrange the patched feature to obtain the image
        x = patched_feature.reshape(
            patched_feature.shape[0], h, w, patch_size, patch_size, 3
        )
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(x.shape[0], 3, h * patch_size, w * patch_size)
        return imgs

    def random_masking(self, x, total_scores):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.

        Args:
            x (torch.Tensor): Input patched feature [N, L, D] with
                            N: Batch numbers
                            L: (H // patch_size) ** 2
                            D: patch_size ** 2 * 3

            total_scores (torch.Tensor): Probability map [N, L]

            self.num_keep_patches (int): Number of patches to keep.

        Returns:
            x_keep (torch.Tensor): Remain patched images after masking [N, L_new, D] with
                                    N: Batch numbers
                                    L_new: self.num_keep_patches
                                    D: patch_size ** 2 * 3

            ids_keep (torch.Tensor):  Ids tensor for corresponding to remain patched feature.
        """
        # Get device of input
        device = x.device
        total_scores = total_scores.to(device)

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(self.num_keep_patches)

        # Sort noise for each sample
        ids_shuffle = torch.multinomial(
            total_scores, num_samples=L, replacement=False
        ).to(device)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep].to(device)
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_keep, ids_keep

    def forward_encoder(self, imgs, total_scores):
        """
        Encoder module of the MCM model.

        Args:
            imgs (torch.Tensor): Batch of images with shape (N, H, W, 3), where
                - N: Number of batches
                - H: Height of the images
                - W: Width of the images

            total_scores (torch.Tensor): Probability map with shape (N, L), where
                - N: Number of batches
                - L: Full patches

        Returns:
            x_keep (torch.Tensor): Remaining patched image with shape (N, L, D), where
                - N: Number of batches
                - L: Number of patches (self.num_keep_patches)
                - D: Dimension of each patch (patch_size ** 2 * 3)

            ids_keep (torch.Tensor): Ids tensor for restoring the original patched feature with shape (N, L), where
                - N: Number of batches
                - L_keep: Keeping patches
        """
        # Embed the input images into patched images
        encoder_imgs = self.encoder_embed(imgs)

        # Add pos_embed without cls_token
        encoder_imgs = encoder_imgs + self.encoder_pos_embed[:, 1:, :]

        # Masking: full_length -> num_keep_patches
        x_keep, ids_keep = self.random_masking(encoder_imgs, total_scores)

        # Append cls_token
        cls_token = self.cls_token + self.encoder_pos_embed[:, :1, :]
        cls_token = cls_token.expand(x_keep.shape[0], -1, -1)
        x_keep = torch.cat((cls_token, x_keep), dim=1)

        # Apply Transformer blocks
        for blk in self.encoder_blocks:
            x_keep = blk(x_keep)
        x_keep = self.encoder_norm(x_keep)

        x_keep = x_keep[:, 1:, :]

        return x_keep, ids_keep

    def forward_decoder(self, x_decode, ids_keep):
        """
        Decoder module of the MCM model.

        Args:
            x_ (torch.Tensor): Remaining patched image with shape (N, L_new, D), where
                - N: Number of batches
                - L_new: Number of patches to keep (self.num_keep_patches)
                - D: Dimension of each patch (patch_size ** 2 * 3)

            ids_restore (torch.Tensor): Ids tensor for restoring the original patched feature with shape (N, L), where
                - N: Number of batches
                - L: Full patches (L = (img_size // patch_size) ** 2)

        Returns:
            patched_imgs (torch.Tensor): Patched reconstruction image with shape (N, L_new, D).
        """
        x_decode = self.decoder_embed(x_decode)
        ids_keep = ids_keep.to(x_decode.device)

        """For training, comment out the code below"""
        noise = torch.rand(
            ids_keep.shape[0], 256, device=x_decode.device
        )  # noise in [0, 1]
        ids_all = torch.argsort(noise, dim=1)
        superset = torch.cat([ids_keep, ids_all], dim=1)
        uniset, count = torch.unique(superset[0], sorted=False, return_counts=True)
        mask = count == 1
        ids_remove = uniset.masked_select(mask).unsqueeze(0)
        for i in range(1, ids_keep.shape[0]):
            uniset, count = torch.unique(superset[i], sorted=False, return_counts=True)
            mask = count == 1
            ids_remove = torch.cat(
                [ids_remove, uniset.masked_select(mask).unsqueeze(0)], dim=0
            )

        ids_restore = torch.cat([ids_keep, ids_remove], dim=1)
        """For training, comment out the code above"""

        ids_restore = torch.argsort(ids_restore, dim=1)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x_decode.shape[0], ids_restore.shape[1] - x_decode.shape[1], 1
        )

        x_ = torch.cat([x_decode, mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_decode.shape[2])
        )  # unshuffle
        x_decode = x_

        # add pos embed
        x_decode = x_decode + self.decoder_pos_embed[:, 1:, :]

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x_decode = blk(x_decode)
        x_decode = self.decoder_norm(x_decode)

        # predictor projection
        x_decode = self.decoder_pred(x_decode)

        return x_decode

    def cal_features_loss(self, preds, imgs):
        """
        Calculates the features loss using VGG16

        Args:
            preds (torch.Tensor): Batch-like predictions (N, H, W, 3)
            imgs (torch.Tensor): Batch-like images (N, H, W, 3)

        Returns:
            feature_loss (torch.Tensor): Batch-like loss
        """

        # Define featuremap network VGG16
        vgg_model = feature_network(net_type="vgg16", requires_grad=False)

        # Denormalize the batch-like images
        pred_F2_denorm = de_normalize(preds)
        gt_F2_denorm = de_normalize(imgs)

        # Normalize the batch-like images
        pred_F2_norm = normalize_batch(pred_F2_denorm)
        gt_F2_norm = normalize_batch(gt_F2_denorm)

        # Featuremap after passing into VGG16
        feature_pred_F2 = vgg_model(pred_F2_norm)
        feature_gt_F2 = vgg_model(gt_F2_norm)
        feature_loss = nn.MSELoss()(
            feature_pred_F2.relu2_2, feature_gt_F2.relu2_2
        ) + nn.MSELoss()(feature_pred_F2.relu3_3, feature_gt_F2.relu3_3)
        return feature_loss

    def forward_loss(self, imgs, preds):
        """
        Loss function

        Args:
            imgs (torch.Tensor): Batch-like images (N, 3, H, W)
            preds (torch.Tensor): Batch-like patches (N, L, D) with
                                N: Batch numbers
                                L: (H // patch_size) ** 2
                                D: patch_size ** 2 * 3

        Returns:
            ssim loss, l1_loss, feature_loss
        """
        preds = self.unpatchify(preds)
        ssim = SSIM(
            win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3
        )
        ssim_loss = 1 - ssim(preds, imgs)

        l1_loss = nn.L1Loss()(preds, imgs)
        feature_loss = self.cal_features_loss(preds, imgs)
        return ssim_loss, l1_loss, feature_loss

    def forward(self, imgs, total_scores):
        """
        MAEC model

        Args:
            imgs (torch.Tensor): Batch-like images (N, H, W, 3)
            total_scores (torch.Tensor): Probability map [N, L]

        Returns:
            patched_imgs (torch.Tensor): Patched reconstruction image
        """
        # Encoder
        x_keep, ids_keep = self.forward_encoder(imgs, total_scores)

        # LIC
        y = (
            x_keep.view(
                -1,
                int(self.num_keep_patches**0.5),
                int(self.num_keep_patches**0.5),
                self.encoder_embed_dim,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        # Apply G_a module
        y = self.g_a(y).float()

        y_shape = y.shape[2:]

        # Apply H_a module
        z = self.h_a(y)

        _, z_likelihood = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = quantize_ste(z_tmp) + z_offset

        # Apply H_s module
        latent_scales = self.h_s_scale(z_hat)
        latent_means = self.h_s_mean(z_hat)

        # Compress using slices
        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihoods = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (
                y_hat_slices
                if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )

            # Calculate mean
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_transform_mean[slice_index](mean_support)
            mu = mu[:, :, : y_shape[0], : y_shape[1]]

            # Calculate scale
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            sigma = self.cc_transform_scale[slice_index](scale_support)
            sigma = sigma[:, :, : y_shape[0], : y_shape[1]]

            # Calculate y_slice_likelihood
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, sigma, mu)
            y_likelihoods.append(y_slice_likelihood)

            # Calculate y_hat_slice
            y_hat_slice = quantize_ste(y_slice - mu) + mu

            # Calculate lrp transform
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transform[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihood = torch.cat(y_likelihoods, dim=1)

        # Apply G_s module
        y_hat = self.g_s(y_hat)
        y_hat = (
            y_hat.permute(0, 2, 3, 1)
            .contiguous()
            .view(-1, self.num_keep_patches, self.encoder_embed_dim)
        )

        # Decoder
        preds = self.forward_decoder(y_hat, ids_keep).float()

        loss = self.forward_loss(imgs, preds)
        x_hat = self.unpatchify(preds)

        return {
            "loss": loss,
            "likelihoods": {"y": y_likelihood, "z": z_likelihood},
            "x_hat": x_hat,
        }

    def compress(self, imgs, total_scores):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for autoregressive models. The entropy coder is run sequentially "
                "on GPU "
            )

        # Encoder MCM
        x_keep, ids_keep = self.forward_encoder(imgs, total_scores)

        # LIC
        y = (
            x_keep.view(
                -1,
                int(self.num_keep_patches**0.5),
                int(self.num_keep_patches**0.5),
                self.encoder_embed_dim,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        # Apply G_a module
        y = self.g_a(y).float()
        y_shape = y.shape[2:]

        # Apply H_a module
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        # Apply H_s module
        latent_scales = self.h_s_scale(z_hat)
        latent_means = self.h_s_mean(z_hat)

        # Compress using slices
        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []

        # CDF
        cdfs = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        # BufferedRansEncoder Module
        encoder = BufferedRansEncoder()

        # Compress using slices
        y_strings = []
        symbols_list = []
        indexes_list = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (
                y_hat_slices
                if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )

            # Calculate mean
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_transform_mean[slice_index](mean_support)
            mu = mu[:, :, : y_shape[0], : y_shape[1]]

            # Calculate scale
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            sigma = self.cc_transform_scale[slice_index](scale_support)
            sigma = sigma[:, :, : y_shape[0], : y_shape[1]]

            index = self.gaussian_conditional.build_indexes(sigma)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            # Calculate lrp transform
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transform[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdfs, cdf_lengths, offsets
        )

        # Get y_string
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {
            "string": [y_strings, z_strings],
            "shape": z.size()[-2:],
            "ids_keep": ids_keep,
        }

    def decompress(self, strings, shape, ids_restore=None):
        assert isinstance(strings, list) and len(strings) == 2

        # Decompress
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        # Apply h_s module
        latent_scales = self.h_s_scale(z_hat)
        latent_means = self.h_s_mean(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_string = strings[0][0]
        y_hat_slices = []

        # Cdf
        cdfs = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        # Decoder bit stream
        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Decompress using slices
        for slice_index in range(self.num_slices):
            # for slice_index in range(3):
            support_slices = (
                y_hat_slices
                if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )

            # Calculate mean
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_transform_mean[slice_index](mean_support)
            mu = mu[:, :, : y_shape[0], : y_shape[1]]

            # Calculate scale
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            sigma = self.cc_transform_scale[slice_index](scale_support)
            sigma = sigma[:, :, : y_shape[0], : y_shape[1]]

            # Index
            index = self.gaussian_conditional.build_indexes(sigma)

            # Revert string indices
            rv = decoder.decode_stream(
                index.reshape(-1).tolist(), cdfs, cdf_lengths, offsets
            )
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            # Lrp transform
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transform[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)

        # Apply G_s module
        y_hat = self.g_s(y_hat)
        y_hat = (
            y_hat.permute(0, 2, 3, 1)
            .contiguous()
            .view(-1, self.num_keep_patches, self.encoder_embed_dim)
        )

        # Decoder MCM
        x_hat = self.forward_decoder(y_hat, ids_restore).float()
        x_hat = self.unpatchify(x_hat)

        return {"x_hat": x_hat}


def MAEC_base_patch16_512(**kwargs):
    model = MAEC(
        img_size=256,
        patch_size=16,
        in_chans=3,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_pix_loss=False,
        latent_depth=384,
        hyperprior_depth=192,
        num_slices=12,
        num_keep_patches=144,
        **kwargs,
    )
    return model


def MAEC_large_patch16_1024(**kwargs):
    model = MAEC(
        img_size=224,
        patch_size=16,
        in_chans=3,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_pix_loss=False,
        latent_depth=384,
        hyperprior_depth=192,
        num_slices=12,
        num_keep_patches=144,
        **kwargs,
    )
    return model


maec_base_patch16 = MAEC_base_patch16_512
maec_large_patch16 = MAEC_large_patch16_1024
