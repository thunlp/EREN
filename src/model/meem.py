from typing import Optional, Iterable
import random

import torch
from torch import nn, Tensor
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput


class EntityEmbedding(nn.Module):
    """
    A memory of entity embeddings that turns entity IDs into embeddings.
    Basically just a wrapper around an embedding and a reparameterizer.
    """

    def __init__(
        self,
        num_ents: int,
        d_out: int,
        do_reparameterize: bool,
        d_mem: Optional[int] = None,
        d_hidden: Optional[int] = None,
    ):
        """
        Args:
            num_ents: Total number of entities in the memory
            d_out: Dimension of the embeddings.
            do_repameterize: Whether to reparameterize the entity embeddings.
        If false, `d_mem` and `d_hidden` will be ignored and output embeddings
        will have `d_out` dimensions.
        """
        super().__init__()
        if do_reparameterize:
            assert d_mem is not None
            assert d_hidden is not None
        else:
            assert d_mem is None
            assert d_hidden is None
        self.num_ents = num_ents
        self.do_reparameterize = do_reparameterize

        self.d_mem = d_out
        self.reparameterizer = nn.Identity()
        if do_reparameterize:
            assert d_mem is not None
            assert d_hidden is not None
            self.d_mem = d_mem
            self.d_hidden = d_hidden
            self.d_out = d_out
            self.reparameterizer = nn.Sequential(
                nn.Linear(d_mem, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_out),
            )
        self.embeddings = nn.Embedding(num_ents, self.d_mem)

    def init_weights_from_embeddings(
        self,
        base_model_embeddings: nn.Embedding,
        sample_range_lo: int = 0,
        sample_range_hi: int = 1,
    ):
        """
        Each entity embedding is initialized to a random embedding of the
        base model. Sampled uniformly from the range [500, 1000).
        """
        random.seed(0)
        d_emb = base_model_embeddings.weight.data.shape[1]
        assert (
            self.d_mem % d_emb == 0
        ), f"Embedding dimension {d_emb} must divide memory "
        # Sample random embeddings from base model
        embs_per_ents = self.d_mem // d_emb
        indices = random.choices(
            range(sample_range_lo, sample_range_hi),
            k=self.num_ents * embs_per_ents,
        )
        rand_embeds = (
            base_model_embeddings.weight.data[indices].detach().clone()
        )  # (num_ents * embs_per_ent, d_emb)
        rand_embeds = rand_embeds.view(self.num_ents, self.d_mem)

        # Copy weights of the embeddings to the entity memory
        self.embeddings.weight.data = rand_embeds
        self.embeddings.weight.requires_grad = True

    def forward(self, entity_ids: Tensor) -> Tensor:
        """
        entity_ids: 1D or 2D tensor of entity_ids
        Return: (..., d_emb)
        """
        # (b, d_mem)
        embeds = self.embeddings(entity_ids)
        embeds = self.reparameterizer(embeds)  # (b, d_emb)
        return embeds


class InputEntityEmbedding(nn.Module):
    """
    An embedding layer that turns entity IDs into a list of embeddings.
    """
    def __init__(
        self,
        num_ents: int,
        do_reparameterize: bool,
        d_emb: int,
        embs_per_ent: int,
        d_mem: Optional[int] = None,
        d_hidden: Optional[int] = None,
    ):
        """
        d_emb: Dimension of the input embeddings of the base model
        embs_per_ent: Number of embeddings per entity
        d_mem: Dimension of the entity memory
        d_hidden: Dimension of the hidden layer of the reparameterizer
        """
        super().__init__()
        self.d_emb = d_emb
        self.embs_per_ent = embs_per_ent

        self.mem = EntityEmbedding(
            num_ents,
            do_reparameterize=do_reparameterize,
            d_out=d_emb * embs_per_ent,
            d_mem=d_mem,
            d_hidden=d_hidden,
        )

    def init_weights(self, base_model_embeddings: nn.Embedding):
        self.mem.init_weights_from_embeddings(base_model_embeddings)

    def forward(self, entity_ids: Tensor) -> Tensor:
        """
        entity_ids: (b, e)

        Notations:
        b: batch size
        e: number of entities
        d: dimension of the input embeddings of the base model

        Return: (b, num_ents * emb_per_ent, d)
        """
        # (b, e, d_ent)
        bsz, num_ents = entity_ids.shape[:2]
        embeds = self.mem(entity_ids)  # (b, e, d_ent)
        embeds = embeds.view(bsz, num_ents * self.embs_per_ent, self.d_emb)
        return embeds


class Meem(nn.Module):
    """
    Model Editing with Entity Memory (MEEM). Use prompts to
    efficiently inject knowledge about an entity into a
    large pretrained language model.
    """
    def __init__(
        self,
        num_entities: int,
        pretrained_name: str,
        prompt_len: int,
        reparam_prompt: bool,
        d_ent: Optional[int] = None,
        d_reparam_hidden: Optional[int] = None,
        cache_dir: str = "../cache",
    ):
        """
        d_ent: Dimension of entity embeddings
        d_hidden: Dimension of hidden layer in the reparameterizer
        prompt_len: Length of the prompt
        """
        super().__init__()
        self.pretrained_name = pretrained_name
        base_model = (
            T5ForConditionalGeneration.from_pretrained(
                pretrained_name, cache_dir=cache_dir
            )
        )
        if isinstance(base_model, T5ForConditionalGeneration):
            self.base_model = base_model
        else:
            raise NotImplementedError(
                f"pretrained_name {pretrained_name} not supported"
            )
        self.prompt_len = prompt_len
        self.reparam_prompt = reparam_prompt

        self.d_emb = self.base_model.config.d_model
        self.d_reparam_hidden = d_reparam_hidden
        if reparam_prompt and d_ent is None:
            self.d_ent = self.d_emb
        else:
            self.d_ent = d_ent
        self.d_kv = self.base_model.config.d_kv
        self.num_layers = self.base_model.config.num_layers
        self.num_heads = self.base_model.config.num_heads

        # self.ent_mem = PrefixEntityMemory(
        #     do_reparameterize=True,
        #     num_entities=num_entities,
        #     embed_dim=self.embed_dim,
        #     mem_dim=self.embed_dim,
        #     hidden_dim=self.embed_dim,
        #     num_layers=self.num_layers,
        #     num_heads=self.num_heads,
        #     d_kv=self.d_kv,
        # )
        if prompt_len > 0:
            self.ent_mem = InputEntityEmbedding(
                num_ents=num_entities,
                do_reparameterize=reparam_prompt,
                embs_per_ent=self.prompt_len,
                d_mem=self.d_ent,
                d_hidden=self.d_reparam_hidden,
                d_emb=self.d_emb,
            )
            self.ent_mem.init_weights(self.base_model.shared)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_train_params(self, params_name: str) -> Iterable[Tensor]:
        print("Training params:", params_name)
        if params_name in ["all", "ft"]:
            assert self.prompt_len == 0
            for p in self.parameters():
                yield p
        elif params_name == "entity-mem":
            assert self.prompt_len > 0
            for p in self.ent_mem.parameters():
                yield p
        elif params_name.startswith("encoder-"):
            if "flan-t5-" in self.pretrained_name:
                layer_num = int(params_name.split("-")[-1])
                assert 0 <= layer_num < self.num_layers
                param_dict = dict(self.named_parameters())
                for name, param in param_dict.items():
                    params = []
                    if f"encoder.block.{layer_num}" in name:
                        print(name)
                        yield param
            else:
                raise NotImplementedError
        elif params_name.startswith("decoder-"):
            if "flan-t5-" in self.pretrained_name:
                layer_num = int(params_name.split("-")[-1])
                assert 0 <= layer_num < self.num_layers
                param_dict = dict(self.named_parameters())
                for name, param in param_dict.items():
                    params = []
                    if f"decoder.block.{layer_num}" in name:
                        yield param
                return params  # type: ignore
            else:
                raise NotImplementedError
        else:
            raise ValueError(f"Invalid params_name: {params_name}")

    def prepare_kp_inputs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        ent_ids: Optional[Tensor] = None,
    ):
        embeds = self.base_model.shared(input_ids)  # (b, s, d)
        if self.prompt_len > 0 and ent_ids is not None:
            # Use knowledge prompts
            ent_ids = ent_ids.unsqueeze(1)  # (b, 1)
            # Concatenate entity and token embeddings
            kp = self.ent_mem(ent_ids)  # (b, # ent embs, d)
            _, num_ent_embs, _ = kp.shape
            embeds = torch.cat([kp, embeds], dim=1)

            # Add 1 on the left for entity
            batch_size, _ = input_ids.shape
            prefix_attention_mask = torch.ones(batch_size, num_ent_embs).to(
                input_ids.device
            )
            attention_mask = torch.cat(
                (prefix_attention_mask, attention_mask), dim=1
            )  # (b, s)
            return embeds, attention_mask
        else:
            # Do not prepend any prompts
            return embeds, attention_mask

    def get_entity_embeds(
        self,
        ent_ids: Tensor,  # (b,)
    ):
        ent_ids = ent_ids.unsqueeze(1)  # (b, 1)

        # Get entity embeddings
        past_key_values = self.ent_mem(ent_ids)  # (b, 1, l*2*d)
        batch_size, _, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            1,
            self.num_layers * 2,
            self.num_heads,
            self.d_kv,
        )  # (b, 1, l*2, #head, head_dim)

        # (2*l, b, #head, 1, head_dim)
        past_key_values = past_key_values.permute(2, 0, 3, 1, 4)
        # tuple: l * (2, b, #head, 1, head_dim)
        past_key_values = past_key_values.split(2)
        return past_key_values

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        ent_ids: Tensor,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        """
        The returned attributes are the same as those returned by the
        `T5ForConditionalGeneration.forward()` method.

        input_ids: (b, s)
        attention_mask: (b, s)
        ent_ids: (b,)
        labels: (b, label_len)

        Notations:
        L: number of layers
        """

        # Prefix
        # A L-tuple of (2, b, #head, 1, head_dim)
        # past_key_values = self.get_entity_embeds(ent_ids, attention_mask)

        # KP
        embeds, attention_mask = self.prepare_kp_inputs(
            input_ids, attention_mask, ent_ids
        )  # (b, s, d), (b, s)

        # Pass to base model
        outputs: Seq2SeqLMOutput = self.base_model(
            # input_ids,
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            # past_key_values=past_key_values,
            **kwargs,
        )
        return outputs

    def generate(
        self,
        input_ids,
        attention_mask,
        ent_ids=None,
        **kwargs,
    ):
        embeds, attention_mask = self.prepare_kp_inputs(
            input_ids,
            attention_mask,
            ent_ids,
        )
        return self.base_model.generate(
            attention_mask=attention_mask,
            inputs_embeds=embeds,
            do_sample=False,
            **kwargs,
        )
