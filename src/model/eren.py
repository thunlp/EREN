from typing import List, Union, Iterable

import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from torch import nn

from .editor import Editor
from .serac import load_baseless_serac
from .embedder import Contriever
from .retriever import MipsRetriever, Retriever
from .utils import gen_texts_mrc, gen_texts_qa, gen_texts_nli


LOCAL_FILES_ONLY = False


class Eren(Editor, nn.Module):
    """
    Main method. Perform edits by reframing the task as a MRC problem in which
    the context may not contain the answer, where
    edits are given as the context in MRC.

    Is an `nn.Module` so that it can be moved to GPU.
    """

    def __init__(
        self,
        model: T5ForConditionalGeneration,
        tokenizer: T5TokenizerFast,
        retriever: Retriever,
        task_type: str,
        ans_options: List[str],
        num_context_examples: int,
        one_step_mrc: bool,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.task_type = task_type
        self.num_context_examples = num_context_examples
        self.ans_options = ans_options
        self.one_step_mrc = one_step_mrc

        if ans_options is not None:
            self.max_length = 10
        else:
            self.max_length = 20

        self.notes_retriever = retriever
        self.notes: List[str] = []

        self.model.eval()

    def edit_by_statements(self, statements: List[str]):
        """Make multiple edits simultaneously for speedup."""
        print(f"Applying {len(statements)} edits...")
        processed = []
        for statement in statements:
            if statement.endswith("."):
                statement = statement[:-1]
            processed.append(statement)
        self.notes.extend(processed)
        self.notes_retriever.add_to_index(processed)  # (n, d)

    def edit_by_examples(self, inputs: List[str], outputs: List[str]):
        """Most existing works apply edits using input-output pairs."""
        raise NotImplementedError

    def create_context(self, context_idxs: Iterable) -> str:
        context_statements = [self.notes[i] for i in context_idxs]
        context = ".\n".join(context_statements) + "."
        # Join as a list
        # context_statements = ["- " + s for s in context_statements]
        # context = "It is known that: " + " ".join(context_statements)
        return context

    def gen_one(
        self, question: str, context_idxs: Union[List[int], torch.Tensor]
    ) -> str:
        # Add context
        context = self.create_context(context_idxs)
        mrc_output = gen_texts_mrc(self.model, self.tokenizer, [context], [question])[0]
        if mrc_output.strip().lower() == "unanswerable":
            return self.gen_qa_one(question)
        else:
            return mrc_output

    def gen_qa_one(self, question: str) -> str:
        return gen_texts_qa(self.model, self.tokenizer, [question])[0]

    def gen_qa(self, questions: List[str]) -> List[str]:
        return [self.gen_qa_one(question) for question in questions]

    def gen_texts(self, questions: List[str]) -> List[str]:
        if len(questions) == 0:
            return []

        if not self.notes:  # No edits have been made.
            raise ValueError("No edits have been made yet!")
            # return self.gen_qa(questions)

        _, ctx_idxs = self.notes_retriever.retrieve(
            questions, topk=self.num_context_examples
        )  # type: ignore

        # Need to evaluate each input separately during test generation.
        contexts = [self.create_context(idxs) for idxs in ctx_idxs]
        if self.task_type == "mrc":
            all_output_texts = gen_texts_mrc(
                self.model,
                self.tokenizer,
                contexts,
                questions,
                options=self.ans_options,
                max_length=self.max_length,
                maybe_unanswerable=not self.one_step_mrc,
            )
        elif self.task_type == "nli":
            if self.one_step_mrc:
                options = ["Yes", "No"]
            else:
                # Two-step MRC with "unanswerable" options
                options = self.ans_options
            all_output_texts = gen_texts_nli(
                self.model,
                self.tokenizer,
                contexts=contexts,
                hypotheses=questions,
                options=options,
                max_length=self.max_length,
            )
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
        return all_output_texts


def load_eren_with_serac(
    base_model_name: str,
    serac_ckpt_path: str,
    task_type: str,
    ans_options: List[str],
    num_context_examples: int,
) -> Eren:
    baseless_serac = load_baseless_serac(serac_ckpt_path)
    # Remove the small counterfactual model
    print("Deleting the small counterfactual model...")
    baseless_serac.__delattr__("replacement")
    baseless_serac.__delattr__("replacement_tok")
    print("Loading base model...")
    model = T5ForConditionalGeneration.from_pretrained(
        base_model_name, local_files_only=LOCAL_FILES_ONLY
    )
    tokenizer = T5TokenizerFast.from_pretrained(
        base_model_name, local_files_only=LOCAL_FILES_ONLY
    )
    if isinstance(model, T5ForConditionalGeneration):
        return Eren(
            model,
            tokenizer,
            baseless_serac,
            task_type,
            ans_options,
            num_context_examples,
        )
    else:
        raise ValueError


def load_eren_contriever(
    base_model_name: str,
    task_type: str,
    ans_options: List[str],
    one_step_mrc: bool,
    num_context_examples: int,
) -> Eren:
    print("Loading contriever")
    embedder = Contriever()
    retriever = MipsRetriever(embedder)
    print("Loading base model...")
    model = T5ForConditionalGeneration.from_pretrained(base_model_name)
    tokenizer = T5TokenizerFast.from_pretrained(base_model_name)
    if isinstance(model, T5ForConditionalGeneration):
        return Eren(
            model,
            tokenizer,
            retriever,
            task_type,
            ans_options=ans_options,
            num_context_examples=num_context_examples,
            one_step_mrc=one_step_mrc,
        )
    else:
        raise ValueError
