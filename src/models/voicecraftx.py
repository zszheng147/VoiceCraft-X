from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, DynamicCache, AutoConfig


def top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, 
    filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = False
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


def min_p_filtering(
    logits, 
    min_p=0.1, # Pro tip: In practice, LLMs use `min_p` in the 0.01-0.2 range.
    filter_value=-float("Inf"), 
    min_tokens_to_keep=1
):
    probs = torch.softmax(logits, dim=-1)
    # Get the probability of the top token for each sequence in the batch
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    # Calculate the actual min_p threshold by scaling min_p with the top token's probability
    scaled_min_p = min_p * top_probs
    # Create a mask for tokens that have a probability less than the scaled min_p
    tokens_to_remove = probs < scaled_min_p

    sorted_indices = torch.argsort(logits, descending=True, dim=-1)
    sorted_indices_to_remove = torch.gather(tokens_to_remove, dim=-1, index=sorted_indices)
    # Keep at least min_tokens_to_keep
    sorted_indices_to_remove[..., : min_tokens_to_keep] = False

    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    return scores_processed


def sampling(logits, min_p=0.0, top_k=50, top_p=0.9, temperature=0.6):
    # Temperature (higher temperature => more likely to sample low probability tokens)
    if temperature != 1.0:
        logits = logits / temperature
    
    if min_p != 0.0:
        logits = min_p_filtering(logits, min_p=min_p)
    else:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p) # Top-p/top-k filtering
    # logits = min_p_filtering(logits, min_p=min_p) # min_p filtering 
    # Sample
    
    if logits.dim() == 2:
        tokens = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    elif logits.dim() == 3:
        batch_size, seq_len, vocab_size = logits.shape
        probs = F.softmax(logits, dim=-1)
        probs_2d = probs.reshape(-1, vocab_size)
        sampled = torch.multinomial(probs_2d, num_samples=1)
        tokens = sampled.reshape(batch_size, seq_len, 1)
    
    return tokens.squeeze(-1)


class VoiceCraftX(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.llm_input_size = config.llm_input_size
        self.llm_output_size = config.llm_output_size
        self.llm_padding_idx = config.llm_padding_idx

        self.num_codebooks = config.num_codebooks
        self.speech_token_size = config.speech_token_size
        self.speech_empty_idx = config.speech_empty_idx
        self.speech_mask_idx = config.speech_mask_idx
        self.speech_eos_idx = config.speech_eos_idx

        silence_tokens = config.silence_tokens
        silence_nums = config.silence_nums

        self.register_buffer(
            "speech_empty", torch.full(
                (self.num_codebooks, 1), self.speech_empty_idx, dtype=torch.long
            )
        )
        self.register_buffer(
            "speech_mask", torch.full(
                (self.num_codebooks, 1), self.speech_mask_idx, dtype=torch.long
            )
        )
        self.register_buffer(
            "speech_eos", torch.full(
                (self.num_codebooks, 1), self.speech_eos_idx, dtype=torch.long
            )
        )
        self.register_buffer(
            "silence_tokens", torch.tensor(silence_tokens).view(
                len(silence_tokens), 1).expand(len(silence_tokens), silence_nums
            ).clone()
        )

        self.speech_num_special_tokens = config.speech_num_special_tokens

        llm_config = AutoConfig.from_pretrained(config.config_path)
        llm_config.attn_implementation = config.attn_implementation
        llm_config.torch_dtype = config.torch_dtype
        self.llm = AutoModelForCausalLM.from_config(llm_config)
        del self.llm.lm_head

        self.llm.train()
        self.mask_embedding = nn.Parameter(torch.randn(1, self.llm_input_size))
        self.speech_embedding = nn.ModuleList([nn.Embedding(
                self.speech_token_size + self.speech_num_special_tokens, 
                self.llm_input_size
            ) for _ in range(self.num_codebooks)
        ])

        self.llm_decoder = nn.ModuleList([ 
            nn.Linear(self.llm_output_size, self.speech_token_size + self.speech_num_special_tokens, bias=False) 
            for _ in range(self.num_codebooks)
        ])

        for i in range(self.num_codebooks):
            self.llm_decoder[i].weight = self.speech_embedding[i].weight

        self.spk_embed_affine_layer = nn.Linear(config.speaker_embedding_size, self.llm_input_size)

    @property
    def device(self):
        return next(self.parameters()).device
    
    def shift_seqs(self, seqs):
        shift_per_row = torch.arange(1, self.num_codebooks + 1, device=self.device).unsqueeze(1)  # [n_codebooks, 1]
        max_shift = shift_per_row.max().item()

        shifted_seqs = []

        for idx, sample in enumerate(seqs):
            seq_len = sample.shape[-1]
            shifted_length = seq_len + max_shift 
            shifted_ids = torch.full((self.num_codebooks, shifted_length), self.config.speech_empty_idx, device=self.device)

            positions = shift_per_row + torch.arange(seq_len, device=self.device).unsqueeze(0)  # [batch_size, seq_len]
            rows = torch.arange(self.num_codebooks, device=self.device).unsqueeze(1).expand(-1, seq_len)  # [batch_size, seq_len]

            shifted_ids[rows, positions] = sample 
            shifted_seqs.append(shifted_ids)

        return shifted_seqs # , patterns
    
    @torch.inference_mode()
    def generate(
        self,
        n_samples: int,
        text: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        speaker_emb: torch.Tensor,
        use_cache: bool = True,
        generation_config: Optional[dict] = None,
        **kwargs
    ):
        text = self.llm.model.embed_tokens(text)
        
        # how to deal with tokens at the end
        prompt_speech = self.shift_seqs(prompt_speech_token)
        prompt_speech = [speech[:, :-3] for speech in prompt_speech]

        # TODO: support batch
        prompt_speech = prompt_speech[0]
        prompt_speech = torch.stack(
            [self.speech_embedding[k](prompt_speech[k]) for k in range(self.num_codebooks)], dim=0
        ).sum(dim=0, keepdim=True)
        
        speaker_emb = self.spk_embed_affine_layer(F.normalize(speaker_emb, dim=-1)).view(1, 1, -1)
        
        hidden_states = torch.cat([text, speaker_emb, prompt_speech], dim=1)
        hidden_states = hidden_states.repeat(n_samples, 1, 1)

        out_tokens = [[] for _ in range(n_samples)]
        stop_len = [0 for _ in range(n_samples)]
        cache = DynamicCache()
        for _ in range(generation_config["max_length"]):
            attention_mask = torch.tril(
                torch.ones((hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[1]), device=hidden_states.device)
            ).to(torch.bool)
            
            outputs = self.llm.model(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask[:, -1, :],
                output_hidden_states=True,
                return_dict=True,
                use_cache=use_cache,
                past_key_values=cache
            )
            hidden_states = outputs.last_hidden_state[:, -1:]
            cache = outputs.past_key_values
            
            next_token_logits = torch.cat([self.llm_decoder[i](hidden_states) for i in range(self.num_codebooks)], dim=1)
            next_token_logits[:, :, self.speech_empty_idx] = -float("Inf")
            next_token_logits[:, :, self.speech_mask_idx] = -float("Inf")

            next_tokens = sampling(
                next_token_logits, 
                min_p=generation_config["min_p"], 
                top_k=generation_config["top_k"], 
                top_p=generation_config["top_p"], 
                temperature=generation_config["temperature"]
            )
            for idx, next_token in enumerate(next_tokens):
                if stop_len[idx] == 0:
                    if self.speech_eos_idx in next_token:
                        stop_len[idx] = len(out_tokens[idx])
                    else:
                        out_tokens[idx].append(next_token)
            if all(stop_len):
                break

            next_speech_token = torch.stack(
                [self.speech_embedding[k](next_tokens[:, k]) for k in range(self.num_codebooks)], dim=1
            ).sum(dim=1, keepdim=True)
            hidden_states = next_speech_token

        unshifted_spans = []
        for out_token in out_tokens:
            span = torch.stack(out_token, dim=0).squeeze(1).transpose(0, 1) # [K T]
            unshifted_span = [span[j, j:- (self.num_codebooks - j)] for j in range(self.num_codebooks)]
            unshifted_span = torch.stack(unshifted_span, dim=0)
            unshifted_span = unshifted_span[:, unshifted_span[0] != 11]
            unshifted_spans.append(unshifted_span)

        return unshifted_spans