""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch.nn.functional as F
import torch
from onmt.modules.util_class import Cast
from onmt.utils.misc import sequence_mask
from onmt.modules.global_attention import SelfAttention


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(src, lengths)
        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        copy_mask = src[:, :, 2]
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths, copy_mask=copy_mask)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

class SNILayer(nn.Module):
    def __init__(self, opt):
        super(SNILayer, self).__init__()
        self.linear_sni = nn.Linear(opt.enc_rnn_size, opt.enc_rnn_size)  # bidirection rnn
        self.linear_out = nn.Linear(opt.enc_rnn_size, 3)  # bidirection rnn
        self.softmax = nn.LogSoftmax(dim=-1)
        self.self_attention = SelfAttention(opt.enc_rnn_size)

    def forward(self, memory_bank, lengths):

        global_info = self.self_attention(memory_bank, lengths)

        states = memory_bank * global_info.unsqueeze(0)

        states = F.relu(self.linear_sni(states))

        states = states.view(-1, states.shape[2])
        out = self.linear_out(states)
        Cast(torch.float32)
        out = self.softmax(out)
        return out

class SNIModel(nn.Module):
    def __init__(self, encoder, decoder, opt):
        super(SNIModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sni = SNILayer(opt)


    def forward(self, src, tgt, lengths, bptt=False):
        enc_state, memory_bank, lengths = self.encoder(src, lengths)  # has dropout in encoder

        tgt = tgt[:-1]  # exclude last target from inputs
        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        if src.size(-1) > 2:
            copy_mask = src[:, :, 2]
        else:
            copy_mask = None
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths, copy_mask=copy_mask)

        #multi task
        if self.training:
            sin_out = self.sni(memory_bank, lengths)
            return dec_out, attns, sin_out
        else:
            return dec_out, attns

