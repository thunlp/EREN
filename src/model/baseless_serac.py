from typing import List, Dict
import copy

import torch
from transformers import (
    DistilBertModel,
    DistilBertTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from torch import nn, Tensor

from .utils import set_dropout, add_padding, add_sep


class BaselessSerac(nn.Module):
    """
    Class for loading SERAC checkpoints from the official repository,
    see http://www.github.com/eric-mitchell/serac.
    """

    def __init__(
        self,
        classifier: DistilBertModel,
        classifier_tok: DistilBertTokenizer,
        replacement: T5ForConditionalGeneration,
        replacement_tok: T5Tokenizer,
        cache_inputs: List[str],
        cache_outputs: List[str],
        cache_embeds: List[Tensor],
    ):
        super().__init__()
        self.classifier, self.classifier_tok = classifier, classifier_tok
        self.replacement, self.replacement_tok = replacement, replacement_tok
        self.cache_inputs = copy.deepcopy(cache_inputs)
        self.cache_outputs = copy.deepcopy(cache_outputs)
        self.cache_embeds = copy.deepcopy(cache_embeds)
        self.scale = 1.0
        self.freeze_cntr = False

    def edit(self, input_text: str, output_text: str):
        """
        To apply an edit, just add it to cache is enough. But we precompute
        the embedding of each edit to speed up the similarity computation
        during the forward pass.
        """
        self.cache_inputs.append(input_text)
        self.cache_outputs.append(output_text)

        # Update cache_embeds
        cls_embeds = self.embed([input_text])  # (1, 1, d)
        self.cache_embeds.append(cls_embeds.view(-1))

    def edit_many(self, input_texts: List[str], output_texts: List[str]):
        """
        Apply many edits for speed.
        """
        processed = []
        for text in input_texts:
            if text.endswith('.'):
                text = text[:-1]
            processed.append(text)
        self.cache_inputs += processed
        self.cache_outputs += output_texts

        # Update cached embeddings
        cls_embeds = self.embed(input_texts)  # (n, d)
        self.cache_embeds += list(cls_embeds)

    def embed(self, input_texts: List[str]) -> Tensor:
        """Return embeddings of input_texts, shape is (n, d)"""
        # print(input_texts)
        input_encodings = self.classifier_tok(
            input_texts, return_tensors="pt", padding="longest"
        )
        input_encodings = input_encodings.to(self.classifier.device)
        with torch.no_grad():
            input_embeds: Tensor = self.classifier(**input_encodings).last_hidden_state[
                :, 0
            ]  # (n, d)
        return input_embeds

    def embedding_logsim_matrix(self, input_texts: List[str]):
        input_encodings = self.classifier_tok(
            input_texts, return_tensors="pt", padding=True
        ).to(self.classifier.device)
        input_embeds: Tensor = (
            self.classifier(**input_encodings).last_hidden_state[:, 0].unsqueeze(1)
        )
        ctx_embeds = torch.stack(self.cache_embeds, dim=0)
        ctx_embeds = ctx_embeds.view(ctx_embeds.shape[0], 1, -1)
        input_embeds = input_embeds.view(input_embeds.shape[0], 1, -1)

        dists: Tensor = (ctx_embeds[None] - input_embeds[:, None]).norm(
            p=2, dim=-1  # type: ignore
        )
        dists = dists**2
        dists = dists.min(-1).values  # get rid of the dists head dimension
        assert dists.min() >= 0, "Shouldn't have negative distances!"
        cls_logsims = -dists * self.scale
        return cls_logsims

    def crossattend_logsim_matrix(self, cls_ctxs, test_input_texts: list) -> Tensor:
        batch = [
            ctx + self.classifier_tok.sep_token + test
            for test in test_input_texts
            for ctx in cls_ctxs
        ]
        batch_toks = self.classifier_tok(batch, return_tensors="pt", padding=True).to(
            self.classifier.device
        )
        batch_logsims = self.classifier(**batch_toks).logits.log_softmax(-1)[:, 0]
        logsim_matrix = batch_logsims.view(len(test_input_texts), len(cls_ctxs))

        return logsim_matrix

    def build_cf_cache_contexts(self):
        sep = " "
        ctxs = [
            cin + sep + clab + sep
            for cin, clab in zip(self.cache_inputs, self.cache_outputs)
        ]
        return ctxs

    def build_cls_cache_inputs(self):
        sep = self.classifier_tok.sep_token
        inputs = [
            cin + sep + clab + sep
            for cin, clab in zip(self.cache_inputs, self.cache_outputs)
        ]
        return inputs

    def build_cf_input_texts(self, input_texts: List[str], closest_idxs: List[int]):
        assert len(closest_idxs) == len(
            input_texts
        ), "Need one cache idx for each test input"

        cache_contexts = self.build_cf_cache_contexts()
        similar_contexts = [cache_contexts[idx] for idx in closest_idxs]
        cf_input_texts = [ctx + inp for ctx, inp in zip(similar_contexts, input_texts)]
        return cf_input_texts

    def generate_cf(self, input_texts: List[str], closest_idxs: List[int]):
        cf_input_texts = self.build_cf_input_texts(input_texts, closest_idxs)
        inputs = self.replacement_tok(
            cf_input_texts, return_tensors="pt", padding=True
        ).to(self.replacement.device)
        outputs = self.replacement.generate(**inputs)
        if isinstance(outputs, Tensor):
            output_texts = self.replacement_tok.batch_decode(
                outputs, skip_special_tokens=True
            )
            return output_texts
        else:
            raise ValueError("Unexpected type of outputs.")

    def is_cache_empty(self) -> bool:
        return len(self.cache_inputs) == 0

    def forward_cls(self, input_texts: List[str]):
        """
        Returns a tuple of:
        1. similary_scores: 1D Tensor of shape (N), each element corresponds to
        the similarity score of the input text with the most similar cached
        edit.
        2. cache_index: 1D Tensor of shape (N), each element corresponds to
        the index of the most similar cached edit.

        N: length of `input_texts`.
        """
        sim_score, cache_idx, _ = self.run_classifier(input_texts)
        return sim_score, cache_idx

    def run_classifier(self, input_texts: List[str]):
        log_sim_matrix = self.embedding_logsim_matrix(input_texts)

        sims = log_sim_matrix.exp()
        assert sims.max() <= 1, "Similarities shouldn't exceed 1!"
        assert sims.min() >= 0, "Similarities shouldn't be negative!"

        cls_sims, cls_idxs = sims.max(-1)
        return cls_sims, cls_idxs, log_sim_matrix


CLS_PRETRAINED_NAME = "distilbert-base-cased"
CF_PRETRAINED_NAME = "t5-small"
LOCAL_FILES_ONLY = True


def init_cf(dropout=0.0):
    cf_tok = T5Tokenizer.from_pretrained(
        CF_PRETRAINED_NAME, local_files_only=LOCAL_FILES_ONLY)
    cf_model = T5ForConditionalGeneration.from_pretrained(
        CF_PRETRAINED_NAME, local_files_only=LOCAL_FILES_ONLY)
    if isinstance(cf_model, T5ForConditionalGeneration) and isinstance(
        cf_tok, T5Tokenizer
    ):
        if cf_tok.sep_token is None:
            add_sep(cf_tok, cf_model)
        if cf_tok.pad_token is None:
            add_padding(cf_tok, cf_model)
        set_dropout(cf_model, dropout)
        return cf_model, cf_tok
    else:
        raise ValueError(f"cf_model is {type(cf_model)}!")


def init_cls(dropout=0.0):
    cls = DistilBertModel.from_pretrained(
        CLS_PRETRAINED_NAME, local_files_only=LOCAL_FILES_ONLY)
    cls_tok = DistilBertTokenizer.from_pretrained(
        CLS_PRETRAINED_NAME, local_files_only=LOCAL_FILES_ONLY)
    if isinstance(cls, DistilBertModel) and isinstance(cls_tok, DistilBertTokenizer):
        # cls.pooler = None  # we don't need the classification head
        set_dropout(cls, dropout)
        return cls, cls_tok
    else:
        raise ValueError(f"cls is {type(cls)}!")


def new_baseless_serac() -> BaselessSerac:
    classifier, classifier_tok = init_cls()
    replacement, replacement_tok = init_cf()
    return BaselessSerac(
        classifier,
        classifier_tok,
        replacement,
        replacement_tok,
        cache_inputs=[],
        cache_outputs=[],
        cache_embeds=[],
    )


def adapt_state_dict(state_dict: Dict[str, Tensor]):
    """Remove keys that I won't be using"""
    key_map = {
        # "classifier.": "classifier.distilbert.",
    }
    to_add = []
    rm_keys = ["scale", "model_config"]
    for src_key, dst_key in key_map.items():
        for old_name, val in state_dict.items():
            if src_key in old_name:
                new_name = old_name.replace(src_key, dst_key)
                to_add.append((new_name, val))
                rm_keys.append(old_name)
    for rm_key in rm_keys:
        del state_dict[rm_key]
    for new_name, val in to_add:
        state_dict[new_name] = val


def load_baseless_serac(ckpt_path: str) -> BaselessSerac:
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print("Loaded")
    state_dict = ckpt["model"]

    adapt_state_dict(state_dict)
    print("Instantiating new serac")
    serac = new_baseless_serac()
    print("Loading serac state dict (scope classifer and counterfactual model)")
    serac.load_state_dict(state_dict)
    return serac
