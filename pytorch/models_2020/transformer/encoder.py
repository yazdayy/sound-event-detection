import torch
from models_2020.transformer.attention import MultiHeadedAttention
from models_2020.transformer.embedding import PositionalEncoding
from models_2020.transformer.encoder_layer import EncoderLayer
from models_2020.transformer.layer_norm import LayerNorm
from models_2020.transformer.positionwise_feed_forward import PositionwiseFeedForward
from models_2020.transformer.repeat import repeat
from models_2020.transformer.subsampling import Conv2dNoSubsampling, Conv2dSubsampling

# Reference: https://github.com/espnet/espnet/tree/master/espnet/nets/pytorch_backend/transformer


class Encoder(torch.nn.Module):
    """Encoder module
    :param int idim: input dim
    :param argparse.Namespace args: experiment config
    """

    def __init__(
    self,
    idim,
    adim: int = 144,
    dropout_rate: float = 0.1,
    elayers: int = 3,
    eunits: int = 576,
    aheads: int = 4,
    kernel_size: int = 7,
    transformer_input_layer = 'conv2d',
    transformer_attn_dropout_rate = 0.0,
    after_conv = False,
    pos_enc=True):
        super(Encoder, self).__init__()
        if transformer_input_layer == "linear":
            if pos_enc:
                self.input_layer = torch.nn.Sequential(
                    torch.nn.Linear(idim, adim),
                    torch.nn.LayerNorm(adim),
                    torch.nn.Dropout(dropout_rate),
                    torch.nn.ReLU(),
                    PositionalEncoding(adim, dropout_rate),
                )
            else:
                self.input_layer = torch.nn.Sequential(
                    torch.nn.Linear(idim, adim),
                    torch.nn.LayerNorm(adim),
                    torch.nn.Dropout(dropout_rate),
                    torch.nn.ReLU(),
                )

        elif transformer_input_layer == "conv2d":
            self.input_layer = Conv2dSubsampling(idim, adim, dropout_rate)
        elif transformer_input_layer == "conv2d_no":
            self.input_layer = Conv2dNoSubsampling(idim, adim, dropout_rate)
        elif transformer_input_layer == "embed":
            self.input_layer = torch.nn.Sequential(
                torch.nn.Embedding(idim, adim), PositionalEncoding(adim, dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + transformer_input_layer)

        self.encoders = repeat(
            elayers,
            lambda: EncoderLayer(
                adim,
                MultiHeadedAttention(aheads, adim, transformer_attn_dropout_rate),
                PositionwiseFeedForward(adim, eunits, dropout_rate),
                dropout_rate,
                after_conv,
            ),
        )
        self.norm = LayerNorm(adim)

    def forward(self, x, mask=None):
        """Embed positions in tensor
        :param torch.Tensor x: input tensor
        :param torch.Tensor mask: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if isinstance(self.input_layer, Conv2dNoSubsampling):
            x, mask = self.input_layer(x, mask)
        elif isinstance(self.input_layer, Conv2dSubsampling):
            x, mask = self.input_layer(x, mask)
        else:
            x = self.input_layer(x)
        #         x, mask = self.encoders(x, mask)
        #         return x, mask
        x, mask = self.encoders(x, mask)
        return self.norm(x), mask
