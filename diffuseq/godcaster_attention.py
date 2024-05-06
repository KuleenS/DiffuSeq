from typing import Optional, Tuple

import torch
from torch import nn

from transformers.pytorch_utils import prune_linear_layer, find_pruneable_heads_and_indices

from godcaster_self_output import GodCasterSelfOutput
from godcaster_self_attention import GodCasterSelfAttention


ROBERTA_SELF_ATTENTION_CLASSES = {
    "eager": GodCasterSelfAttention
}


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Roberta,BERT->ROBERTA
class RobertaAttention(nn.Module):
    def __init__(self, config, video_shape, position_embedding_type=None):
        super().__init__()
        self.video_shape = video_shape
        self.self = ROBERTA_SELF_ATTENTION_CLASSES[config._attn_implementation](
            config, self.video_shape, position_embedding_type=position_embedding_type
        )
        self.output = GodCasterSelfOutput(config, self.video_shape)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
