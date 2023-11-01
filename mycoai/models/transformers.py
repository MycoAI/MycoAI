'''Implementation of transformer-based models for ITS classification 

Transformer:
    Publication: 'Attention Is All You Need' by Vaswani et al. (2017) 
                 (https://doi.org/10.48550/arXiv.1706.03762).
    Code:         https://nlp.seas.harvard.edu/annotated-transformer/ 
                  by Austin Huang

BERT:
    Publication: 'BERT: Pre-training of Deep Bidirectional Transformers for 
                 Language Understanding' by Devlin et al. (2019)
                 (https://doi.org/10.48550/arXiv.1810.04805)'''

import math
import torch
from .. import utils

class BERT(torch.nn.Module):
    '''BERT base model, transformer encoder to be used for various tasks'''

    def __init__(self, len_input, vocab_size, d_model=512, d_ff=2048, h=8, N=6, 
                 dropout=0.1, mode='default'):
        '''Initializes the transformer given the source/target vocabulary.
        
        Parameters
        ----------
        len_input: int
            Length of expected input sequences
        vocab_size: int
            Number of unique tokens in vocabulary
        d_model: int
            Dimension of sequence repr. (embedding) in model (default is 512)
        d_ff: int
            Dimension of hidden layer FFN sublayers (default is 2048)
        h: int
            Number of heads used for multi-head self-attention (default is 8)
        N: int
            How many encoder/decoder layers the transformer has (default is 6)
        dropout: float
            Dropout probability to use throughout network (default is 0.1)
        mode: str
            BERT will use different forward method when mode=='classification'
        '''

        super().__init__()
        self.src_pos_embed = PositionalEmbedding(d_model, vocab_size, dropout)
        self.encoder = Encoder(d_model, d_ff, h, N, dropout)
        self.mlm_layer = torch.nn.Linear(d_model, vocab_size)
        self.len_input = len_input
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.N = N
        self.set_mode('default')
        self.to(utils.DEVICE)

        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def set_mode(self, mode):
        '''Uses alternative forward method when mode == 'classificiation'.'''
        if mode == 'classification':
            self.forward = self._forward_classification
        elif mode == 'mlm':
            self.forward = self._forward_mlm
        else:
            self.forward = self._forward_default

    def _forward_default(self, src, src_mask):
        '''Given a source (and mask), retrieve encoded representation'''
        src_embedding = self.src_pos_embed(src)
        return self.encoder(src_embedding, src_mask)
    
    def _forward_classification(self, src):
        '''Given input sequence, retrieve embedding aggregated in CLS token'''
        src_mask = (src != utils.TOKENS['PAD']).unsqueeze(-2) # Mask padding
        src_embedding = self.src_pos_embed(src) 
        return self.encoder(src_embedding, src_mask)[:,0,:] # Only CLS token

    def _forward_mlm(self, src):
        '''Given input sequence, predict masked tokens'''
        src_mask = (src != utils.TOKENS['PAD']).unsqueeze(-2) # Mask padding
        src = self.encoder(self.src_pos_embed(src), src_mask)
        src = self.mlm_layer(src)
        return torch.softmax(src, dim=-1)
    
    def get_config(self):
        return {
            'type':    utils.get_type(self),
            'd_model': self.d_model,
            'd_ff':    self.d_ff,
            'h':       self.h,
            'N':       self.N
        }


class Encoder(torch.nn.Module):
    '''N layers of consisting of self-attention and feed forward sublayers,
    gradually transforms input into encoded representation.'''

    def __init__(self, d_model, d_ff, h, N, dropout):
        super().__init__()
        layers = []
        for i in range(N): 
            sublayers = torch.nn.ModuleList([
                MultiHeadAttention(h, d_model, dropout),
                ResidualConnection(d_model, dropout),
                FeedForward(d_model, d_ff, dropout),
                ResidualConnection(d_model, dropout)
            ])
            layers.append(sublayers)
        self.layers = torch.nn.ModuleList(layers)
        self.norm = torch.nn.LayerNorm(d_model) 

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer[1](x, lambda x: layer[0](x, x, x, mask))
            x = layer[3](x, layer[2])
        return self.norm(x)


class MultiHeadAttention(torch.nn.Module):
    ''''Performs scaled dot product attention on h uniquely learned linear 
    projections (allowing model to attend to info from different subspaces)'''

    def __init__(self, h, d_model, dropout):
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model)
                                            for i in range(4)]) #NOTE is 4 best?
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2 from 'Attention Is All You Need'"

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)
    

class FeedForward(torch.nn.Module):
    '''Simple feed forward network (with dropout applied to mid layer)'''

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.lin_1 = torch.nn.Linear(d_model, d_ff)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.lin_2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.relu(self.lin_1(x))
        x = self.dropout(x)
        return self.lin_2(x)


class PositionalEmbedding(torch.nn.Module):
    '''Converts input into sum of a learned embedding and positional encoding'''

    def __init__(self, d_model, vocab, dropout):
        super().__init__()

        self.embedder = torch.nn.Embedding(vocab, d_model)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = d_model

        # Compute the positional encodings once in log space.
        pe = torch.zeros(utils.MAX_LEN, d_model)
        position = torch.arange(0, utils.MAX_LEN).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = self.embedder(x) * math.sqrt(self.d_model) # Get embedding
        x = x + self.pe[:, : x.size(1)].requires_grad_(False) # + positional enc
        return self.dropout(x) # Apply dropout
    

class ResidualConnection(torch.nn.Module):
    '''Employs a normalized residual connection followed by dropout'''

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = torch.nn.LayerNorm(size)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, layer):
        '''Adds layer(x) to x and applies normalization/dropout'''
        return x + self.dropout(layer(self.norm(x)))


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask_value = -1e9 if scores.dtype == torch.float32 else -1e4
        scores = scores.masked_fill(mask == 0, mask_value)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn) 
    return torch.matmul(p_attn, value), p_attn


# class Transformer(torch.nn.Module):
#     '''Transformer encoder/decoder architecture. NOTE: this model is currently
#     not implemented in a way to be integrated into mycoai.'''

#     def __init__(self, src_vocab, tgt_vocab, d_model=512, d_ff=2048, h=8, N=6, 
#                  dropout=0.1, max_len=5000):
#         '''Initializes the transformer given the source/target vocabulary.
        
#         Parameters
#         ----------
#         src_vocab: TODO
#             TODO
#         tgt_vocab: TODO
#             TODO
#         d_model: int
#             Dimension of sequence repr. (embedding) in model (default is 512)
#         d_ff: int
#             Dimension of hidden layer FFN sublayers (default is 2048)
#         h: int
#             Number of heads used for multi-head self-attention (default is 8)
#         N: int
#             How many encoder/decoder layers the transformer has (default is 6)
#         dropout: float
#             Dropout probability to use throughout network (default is 0.1)
#         max_len: int
#             Maximum supported length of input by pos. encoders (default is 5000)
#         '''

#         super().__init__()
#         self.src_pos_embed = PositionalEmbedding(d_model, src_vocab, dropout, max_len)
#         self.encoder = Encoder(d_model, d_ff, h, N, dropout)
#         self.tgt_pos_embed = PositionalEmbedding(d_model, tgt_vocab, dropout, max_len)
#         self.decoder = Decoder(d_model, d_ff, h, N, dropout)
#         self.generator = Generator(d_model, tgt_vocab)

#         # Initialize parameters with Glorot / fan_avg.
#         for p in self.parameters():
#             if p.dim() > 1:
#                 torch.nn.init.xavier_uniform_(p)

#     def forward(self, src, tgt, src_mask, tgt_mask):
#         '''Full pass through transformer (embedding, encoding, decoding)'''
#         memory = self.encode(src, src_mask)
#         return self.decode(memory, src_mask, tgt, tgt_mask)

#     def encode(self, src, src_mask):
#         '''Given a source (and mask), retrieve encoded representation'''
#         src_embedding = self.src_pos_embed(src)
#         return self.encoder(src_embedding, src_mask)

#     def decode(self, encoding, src_mask, tgt, tgt_mask):
#         '''Given an encoding and target (+ masks), retrieve decoded repr.'''
#         tgt_embedding = self.tgt_pos_embed(tgt)
#         return self.decoder(tgt_embedding, encoding, src_mask, tgt_mask)


# class Decoder(torch.nn.Module):
#     '''N layers of consisting of (masked) (self-)attention and FF sublayers,
#     gradually transforms encoder's output and output embedding into decoding'''

#     def __init__(self, d_model, d_ff, h, N, dropout):
#         super().__init__()
#         layers = []
#         for i in range(N): 
#             sublayers = torch.nn.ModuleList([
#                 MultiHeadAttention(h, d_model, dropout), # self attention
#                 ResidualConnection(d_model, dropout),
#                 MultiHeadAttention(h, d_model, dropout), # src attention
#                 ResidualConnection(d_model, dropout),
#                 FeedForward(d_model, d_ff, dropout), # feed forward network
#                 ResidualConnection(d_model, dropout)
#             ])
#             layers.append(sublayers)
#         self.layers = torch.nn.ModuleList(layers)
#         self.norm = torch.nn.LayerNorm(d_model) 

#     def forward(self, x, m, src_mask, tgt_mask):
#         "Pass the input (and mask) through each layer in turn."
#         for layer in self.layers:
#             x = layer[1](x, lambda x: layer[0](x, x, x, tgt_mask)) # self att.
#             x = layer[3](x, lambda x: layer[2](x, m, m, src_mask)) # src att.
#             x = layer[5](x, layer[4])
#         return self.norm(x)


# def subsequent_mask(size): 
#     "Mask out subsequent positions."
#     attn_shape = (1, size, size)
#     subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
#         torch.uint8)
#     return subsequent_mask == 0


# class Generator(torch.nn.Module): 
#     "Define standard linear + softmax generation step."

#     def __init__(self, d_model, vocab):
#         super(Generator, self).__init__()
#         self.proj = torch.nn.Linear(d_model, vocab)

#     def forward(self, x):
#         return torch.nn.functional.log_softmax(self.proj(x), dim=-1)