from torch.nn import TransformerDecoderLayer, TransformerDecoder

class TransformerDecoderLayer(TransformerDecoderLayer):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", norm_first=False, batch_first=True):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, norm_first=norm_first, batch_first=batch_first)


    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2, alignment = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask, average_attn_weights=False)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, alignment
    

class TransformerDecoder(TransformerDecoder):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__(decoder_layer, num_layers, norm)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        alignments = []

        for layer in self.layers:
            output, layer_alignment = layer(output, memory, tgt_mask=tgt_mask,
                                            memory_mask=memory_mask,
                                            tgt_key_padding_mask=tgt_key_padding_mask,
                                            memory_key_padding_mask=memory_key_padding_mask)
            
            alignments.append(layer_alignment)
            
        if self.norm is not None:
            output = self.norm(output)

        return output, alignments