from typing import Optional

from torch import Tensor
from torch.nn import TransformerDecoderLayer, TransformerDecoder


class TransformerDecoderLayer(TransformerDecoderLayer):
    '''
    Custom decoder layer which returns alignments.
    '''
    def __init__(
        self, 
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = False, 
        batch_first: bool = True
    ):
        super().__init__(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            norm_first=norm_first, 
            batch_first=batch_first
        )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = True,
        memory_is_causal: bool = False,
    ):
        x, alignments = tgt, None
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x_, alignments = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + x_
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x_, alignments = self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = self.norm2(x + x_)
            x = self.norm3(x + self._ff_block(x))

        return x, alignments
    
    def _mha_block(
        self, 
        x: Tensor, 
        mem: Tensor,
        attn_mask: Optional[Tensor], 
        key_padding_mask: Optional[Tensor], 
        is_causal: bool = False
    ) -> Tensor:
        '''
        x                   :(#bs, #frame, #hidden)
        mem                 :(#bs, #phon,  #hidden)
        attn_mask           :(#bs, #frame, #phon) or None
        key_padding_mask    :(#bs, #frame, 1) or None
        '''
        x, alignments = self.multihead_attn(x, mem, mem,
                                            attn_mask=attn_mask,
                                            key_padding_mask=key_padding_mask,
                                            is_causal=is_causal,
                                            need_weights=True,
                                            average_attn_weights=False)
        return self.dropout2(x), alignments


class TransformerDecoder(TransformerDecoder):
    '''
    Custom decoder layer wrapper which returns alignments from each layer.
    '''

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__(decoder_layer, num_layers, norm)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: Optional[bool] = None
    ):
        alignments = []
        for layer in self.layers:
            tgt, alignment = layer(tgt=tgt,
                                    memory=memory,
                                    tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    tgt_is_causal=tgt_is_causal,
                                    memory_is_causal=memory_is_causal)
            alignments.append(alignment)

        if self.norm is not None:
            tgt = self.norm(tgt)

        return tgt, alignments