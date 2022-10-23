import timm
import torch.nn as nn


class ViTEncoder(nn.Module):
    """ The ViT encoder.

    The Vision Transformer (ViT) model is made up of a patch embedding layer and a
    transformer encoder. The input image is split into patches, which are then projected
    into a lower-dimensional space using a linear layer. The patch embeddings are then
    combined with positional embeddings, and an extra [cls] token is added to the start
    of the sequence. The transformer encoder then processes the sequence of embeddings
    and outputs a sequence of encoded features. The encoded features are finally passed
    through a MLP head to produce a feature vector.

    This network a pretrained ViT model and replaces the head with a linear layer with 784
    output features. The features are reshaped to (N, 16, 7, 7) where N is the batch
    size. The patch embeddings are also returned without the [cls] token.

    By default, the ViT model is 'vit_tiny_patch16_224'. The ViT model can be changed
    by passing the name of the model as an argument. For example, to use the ViT-Small
    model, pass 'vit_small_patch16_224' as the name argument. 16 is the patch size and
    224 is the image size. The ViT models can be found here:
    https://rwightman.github.io/pytorch-image-models/models/vision-transformer/

    Args:
        name (str, optional): The name of the ViT model. Defaults to 'vit_small_patch16_224'.
    """

    def __init__(self, name='vit_tiny_patch16_224'):
        super().__init__()
        self.model = timm.create_model(name, pretrained=True)
        self.model.eval()
        self.model.head = nn.Linear(in_features=self.model.head.in_features, out_features=784, bias=True)

    def forward(self, x):
        """ Forward pass.

        Args:
            x (torch.Tensor): The input Tensor. Shape (B, 3, 224, 224).

        Returns:
            torch.Tensor: The output Tensor. Shape (B, 16, 7, 7).
        """
        B = x.shape[0]

        # Get patch embeddings
        patch_embeddings = self.model.forward_features(x)

        # Forward pass through the MLP head and reshape to (B, 16, 7, 7)
        x = self.model.forward_head(patch_embeddings)
        x = x.view((B, 16, 7, 7))

        # Return encoded features and patch embeddings (without the [cls] token)
        return x, patch_embeddings[:, 1:]

