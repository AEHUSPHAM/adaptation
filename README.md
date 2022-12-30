# Changelog
modeling_roberta.py
## Mask matrix
```python
def momentum_matrix(n=10,m=0.9,s=0.1):
    tensor=torch.Tensor([s*pow(m,i) for i in range(0,n)])
    mask_matrix=torch.cat([F.pad(tensor,(i,0),"constant",0)[:-i] if i>0 else F.pad(tensor,(i,0),"constant",0) for i in range(0,n)],dim=0).view(-1,n)
    mask_matrix = torch.transpose(mask_matrix,0,1)
    return mask_matrix
```
## Modification
```python
class RobertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.config = config

        if config.attn_mode == "adapter" and config.attn_option == "sequential":
            self.ef_attn_adapter = Adapter_Layer(d_model=config.hidden_size,
                                                 dropout=config.attention_probs_dropout_prob,
                                                 bottleneck=self.config.attn_bn,
                                                 adapter_layernorm_option="in",
                                                 )
            # Adding
            self.momentum_matrix = momentum_matrix(config.max_position_embeddings-2)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        if self.config.attn_mode == "adapter" and self.config.attn_option == "sequential":
            hidden_states = self.ef_attn_adapter(hidden_states, add_residual=True)

        hidden_states = self.dropout(hidden_states)
        # Old: hidden_states = self.LayerNorm(torch.matmul(hidden_states + input_tensor)
        hidden_states = self.LayerNorm(torch.matmul(self.momentum_matrix.to(hidden_states.device),hidden_states) + input_tensor)
        return hidden_states
```
```python
class RobertaOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.ffn_mode == 'lora':
            self.dense = Linear(config.intermediate_size, config.hidden_size, r=config.ffn_bn, lora_alpha=config.lora_alpha,
                              lora_dropout=config.lora_dropout, lora_init=config.lora_init)
        else:
            self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.config = config

        if config.ffn_mode == 'adapter':
            self.ef_ffn_adapter = Adapter_Layer(d_model=config.hidden_size,
                                                dropout=config.hidden_dropout_prob,
                                                bottleneck=config.ffn_bn,
                                                init_option=config.ffn_adapter_init_option,
                                                adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                                adapter_scalar=config.ffn_adapter_scalar,
                                                )
            # Adding
            self.momentum_matrix = momentum_matrix(config.max_position_embeddings-2)

    def forward(self, hidden_states, input_tensor, adapter_change=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.config.ffn_mode == 'adapter':
            if self.config.ffn_option == 'sequential':
                hidden_states = self.ef_ffn_adapter(hidden_states)
            elif self.config.ffn_option == 'parallel' and adapter_change is not None:
                hidden_states =  hidden_states + adapter_change

        if self.config.ffn_mode == 'adapter' and self.config.ffn_option == 'pfeiffer':
            h_before_residule = hidden_states
        
        # Old: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = self.LayerNorm(torch.matmul(self.momentum_matrix.to(hidden_states.device),hidden_states) + input_tensor)
        if self.config.ffn_mode == 'adapter' and self.config.ffn_option == 'pfeiffer':
            hidden_states = self.ef_ffn_adapter(hidden_states, residual=h_before_residule)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
```