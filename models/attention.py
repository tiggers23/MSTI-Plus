import torch.nn as nn
import torch
import math

class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, ctx_dim=None):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim = hidden_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)


        # Apply the attention mask is
        if attention_mask is not None:

            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class AttentionOutput(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(AttentionOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states) #非线性变换
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size, cross_size1, nhead=1, dropout=0.1):
        # q:hidden_size
        # k,v:cross_size1
        super(CrossAttentionLayer, self).__init__()
        self.cross_attention_1 = Attention(hidden_size, nhead, dropout, ctx_dim=cross_size1)
        # self.cross_attention_2 = Attention(hidden_size, nhead, dropout, ctx_dim=cross_size2)
        self.self_attention = Attention(hidden_size, nhead, dropout)

        self.out1 = AttentionOutput(hidden_size, dropout)
        self.out2 = AttentionOutput(hidden_size, dropout)
        # self.out3 = AttentionOutput(hidden_size, dropout)

        # self.gate = Gate(hidden_size, hidden_size)

    def forward(self, hidden_states, cross_states_1, attention_mask=None):
        cross_1 = self.cross_attention_1(hidden_states, cross_states_1, attention_mask=attention_mask)
        cross_1 = self.out1(cross_1, hidden_states)

        # cross_2 = self.cross_attention_1(hidden_states, cross_states_2, attention_mask=attention_mask)
        # cross_2 = self.out1(cross_2, hidden_states)
        #
        # cross_fusion = self.gate(cross_1, cross_2)
        self_out = self.self_attention(cross_1, cross_1, attention_mask=attention_mask)
        self_out = self.out2(self_out, cross_1)

        return self_out