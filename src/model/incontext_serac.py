from typing import List, Optional

import torch
from torch import Tensor
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
)

from .serac import Serac, load_baseless_serac
from .incontext_el import create_context


def translate_tokens(tokens, from_tok, to_tok):
    tokens = tokens.masked_fill(tokens == -100, from_tok.pad_token_id)
    text = from_tok.batch_decode(tokens, skip_special_tokens=True)
    return to_tok(text, return_tensors="pt")["input_ids"].to(tokens.device)


CLS_PRETRAINED_NAME = "distilbert-base-cased"
CF_PRETRAINED_NAME = "t5-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class IncontextSerac(Serac):
    def forward_cls(
        self,
        input_texts: List[str],
        threshold: Optional[float] = None,
        topk: Optional[int] = None,
    ):
        log_sim_matrix = self.serac.embedding_logsim_matrix(input_texts)

        sims = log_sim_matrix.exp()  # (num_inputs, cache_size)
        assert sims.max() <= 1, "Similarities shouldn't exceed 1!"
        assert sims.min() >= 0, "Similarities shouldn't be negative!"

        assert not all([threshold is not None, topk is not None])

        if topk is not None:
            # Get indices of to-k most similar cached examples.
            cls_sims, cls_idxs = torch.topk(sims, topk, dim=1)
            return cls_sims, cls_idxs
        elif threshold is not None:
            # Get all indices whose similary is above the threshold.
            print(self.serac.cache_inputs[:10])
            print(input_texts)
            print(sims)
            print(sims.shape)
            sim_x, sim_y = torch.where(sims > threshold)
            print(sim_x, sim_y)
            cls_sims = sims[sim_x, sim_y]
            print(cls_sims)
            print(cls_sims.shape)
            exit()
            raise NotImplementedError

    def create_context(self, context_idxs: Tensor):
        context_examples = [
            (self.serac.cache_inputs[i], self.serac.cache_outputs[i])
            for i in context_idxs
        ]
        context = create_context(context_examples, flan_format=True)
        return context

    def gen_texts(self, input_texts: List[str], num_context_examples):
        if self.serac.is_cache_empty():
            return self.gen_texts_base_model(input_texts)

        _, cache_idxs = self.forward_cls(
            input_texts, topk=num_context_examples)  # type: ignore
        num_inputs = len(input_texts)

        # Need to evaluate each input separately during test generation.
        all_output_texts = []
        for input_idx in range(num_inputs):
            input_text = input_texts[input_idx]
            # Add context
            context_idxs = cache_idxs[input_idx]
            context = self.create_context(context_idxs)
            # TODO: Support non-FLAN format.
            input_text = "{} Q: {} A:".format(context, input_text)
            output_texts = self.gen_texts_base_model([input_text])
            all_output_texts += output_texts
        return all_output_texts


def load_incontext_serac(
    base_model_name: str, serac_ckpt_path: str, cache_dir: str
) -> IncontextSerac:
    model = T5ForConditionalGeneration.from_pretrained(
        base_model_name, cache_dir=cache_dir
    )
    tokenizer = T5TokenizerFast.from_pretrained(base_model_name, cache_dir=cache_dir)
    baseless_serac = load_baseless_serac(serac_ckpt_path)
    if isinstance(model, T5ForConditionalGeneration):
        return IncontextSerac(baseless_serac, model, tokenizer)
    else:
        raise ValueError
