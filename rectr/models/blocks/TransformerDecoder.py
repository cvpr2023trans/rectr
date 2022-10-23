import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.models.layers import trunc_normal_

from .TransformerDecoderLayer import TransformerDecoderLayer


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class TransformerDecoder(nn.Module):
    """ The TransformerDecoder Network. 
    
    It takes the patch embeddings of shape (B, 196, 192) from the ViTEncoder and creates
    a mask of shape (B, K, 14, 14) using a series of Transformer layers. The mask is then
    upsampled to (B, K, 224, 224) using bilinear interpolation.
    
    Adapted from Segmenter: Transformer for Semantic Segmentation
    (https://github.com/rstrudel/segmenter).
    """
    def __init__(self, n_cls=2, patch_size=16, d_encoder=192,
                 n_layers=12, n_heads=3, d_model=192, d_ff=768,
                 drop_path_rate=0.1, dropout=0.0):
        """ Initialize the TransformerDecoder Network.

        Args:
            n_cls (int, optional): The number of classes. Defaults to 2 (background and foreground).
            patch_size (int, optional): The size of the patches. Defaults to 16.
            d_encoder (int, optional): The dimension of the patch embeddings. Defaults to 192.
            n_layers (int, optional): The number of Transformer layers. Defaults to 12.
            n_heads (int, optional): The number of heads in the MultiHeadAttention layer. Defaults to 3.
            d_model (int, optional): The dimension of the MultiHeadAttention layer. Defaults to 192.
            d_ff (int, optional): The dimension of the FeedForward layer. Defaults to 768.
            drop_path_rate (float, optional): The drop path rate. Defaults to 0.1.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
        """
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [TransformerDecoderLayer(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))  # (1, 3, 192)
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x, im_size=(224, 224)):
        H, W = im_size
        GS = H // self.patch_size

        # Project the patch embeddings to the dimension of the MultiHeadAttention layer
        # and concatenate the class embeddings to the patch embeddings
        x = self.proj_dec(x)  # (B, 196, 192) -> (B, 196, 192)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)  # (1, K, 192) -> (B, K, 192)
        x = torch.cat((x, cls_emb), 1) # (B, 196, 192) + (B, K, 192) -> (B, 199, 192)

        # Forward pass through the Transformer layers and normalize the output
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)  # (B, 199, 192)

        # Split the output into the patch embeddings and the class embeddings
        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]  # (B, 196, 192), (B, K, 192)

        # Apply the projection matrices to the patch and class embeddings
        patches = patches @ self.proj_patch  # (B, 196, 192) @ (192, 192) -> (B, 196, 192)
        cls_seg_feat = cls_seg_feat @ self.proj_classes  # (B, K, 192) @ (192, 192) -> (B, K, 192)

        # Apply L2 normalization to the patch and class embeddings
        patches = patches / patches.norm(dim=-1, keepdim=True)  # (B, 196, 192)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)  # (B, K, 192)

        # Generate K masks by computing the scalar product between L2-normalized patch embeddings
        # and class embeddings and reshape the masks from (B, (14*14), K) to (B, K, 14, 14)
        masks = patches @ cls_seg_feat.transpose(1, 2)  # (B, 196, 192) @ (B, 192, K) -> (B, 196, K)
        masks = self.mask_norm(masks)  # apply LayerNorm to the mask
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))  # (B, 196, K) -> (B, K, 14, 14)

        # Upsample the masks to the original image size using bilinear interpolation
        masks = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False)  # (B, K, 224, 224)
        masks = F.softmax(masks, dim=1)  # (N, 2, 224, 224)

        return masks

    def get_attention_map(self, x, layer_id):
        """ Get the attention map of a specific layer.

        This function is used to visualize the attention map of a specific layer.

        Args:
            x (Tensor): The input Tensor. Shape: (B, C, H, W).
            layer_id (int): The id of the layer. Range: [0, n_layers - 1].
        """
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)
