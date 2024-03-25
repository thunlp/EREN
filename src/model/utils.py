import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


def set_dropout(model: nn.Module, p: float):
    if p is not None:
        n_reset = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = p
                n_reset += 1

            if hasattr(m, "dropout"):  # Requires for BART, which uses F.dropout
                if isinstance(m.dropout, float):
                    m.dropout = p  # type: ignore
                    n_reset += 1

            if hasattr(
                m, "activation_dropout"
            ):  # Requires for BART, which uses F.dropout
                if isinstance(m.activation_dropout, float):
                    m.activation_dropout = p  # type: ignore
                    n_reset += 1

        print(f"Set {n_reset} dropout modules to p={p}")


def should_shift_target(model_name: str) -> bool:
    return "t5" not in model_name.lower() and "blender" not in model_name.lower()


def add_padding(tokenizer, model):
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
    model.transformer.wte.weight.data[-1] = model.transformer.wte.weight.data.mean(0)


def add_sep(tokenizer, model):
    tokenizer.add_special_tokens({"sep_token": "[SEP]"})


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


TEMPLATE_QA = "Please answer this question: {question}"
TEMPLATE_QA_2 = 'Can we conclude that "{question}"?'
TEMPLATE_MRC = (
    "Read this and answer the question."
    ' If the question is unanswerable, say "unanswerable".'
    "\n\n{context}\n\n{question}"
)
TEMPLATE_NLI = (
    "{context}\n\nBased on the paragraph above can we conclude that "
    '"{hypothesis}"?'
    "\n\n{options}"
)
TEMPLATE_NLI_2 = (
    "{context}\n\nBased on the paragraph above,"
    " can we conclude that \"{hypothesis}\"?\n\n{options}"
)
TEMPLATE_NLI_3 = '{context}\n\nCan we conclude that "{hypothesis}"?\n\n{options}'
TEMPLATE_NLI_4 = "{context}\n\n{hypothesis}, is that right?\n\n{options}"
TEMPLATE_BOOL_QA = "Is it true that {statement}?\n\n{options}"
TEMPLATE_MRC_WITH_OPTIONS = (
    "Read this and answer the question.\n\n{context}\n\n{question}\n\n{options}"
)
TEMPLATE_MRC_ONE_STEP = (
    "Read this and answer the question.\n\n{context}\n\n{question}"
)

# FLAN template for the BoolQ dataset
TEMPLATE_BOOL_Q = "{context}\n\nCan we conclude that {question}?\n\n{options}"


def create_options(options: List[str]) -> str:
    return "OPTIONS:\n- " + "\n- ".join(options)


def gen_texts(model, tokenizer, input_texts: List[str], max_length: int = 20):
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs, max_length=max_length)
    output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return output_texts


def gen_texts_qa(
    model, tokenizer, input_texts: List[str], max_length: int = 20
) -> List[str]:
    input_texts = [TEMPLATE_QA.format(question=question) for question in input_texts]
    output_texts = gen_texts(model, tokenizer, input_texts, max_length=max_length)
    # for q, a in zip(input_texts, output_texts):
    #     print("------")
    #     print(q)
    #     print(a)
    #     exit()
    return output_texts


def gen_texts_mrc(
    model,
    tokenizer,
    contexts: List[str],
    questions: List[str],
    max_length: int = 20,
    options: Optional[List[str]] = None,
    maybe_unanswerable: Optional[bool] = True,
) -> List[str]:
    input_texts: List[str] = []
    if options is None:
        template = TEMPLATE_MRC if maybe_unanswerable else TEMPLATE_MRC_ONE_STEP
        for context, question in zip(contexts, questions):
            input_texts.append(template.format(context=context, question=question))
    else:
        options_str = create_options(options)
        for context, question in zip(contexts, questions):
            # question = "Is it true that Danielle Darrieux's mother tongue is French?"
            input_texts.append(
                TEMPLATE_MRC_WITH_OPTIONS.format(
                    context=context, question=question, options=options_str
                )
            )
    # print(input_texts)
    output_texts = gen_texts(model, tokenizer, input_texts, max_length=max_length)
    # for q, a in zip(input_texts, output_texts):
    #     print("------")
    #     print(q)
    #     print(a)
    #     exit()
    return output_texts


def gen_texts_nli(
    model,
    tokenizer,
    contexts: List[str],
    hypotheses: List[str],
    options: List[str],
    max_length: int = 20,
) -> List[str]:
    """
    Ask whether a context can be used to infer a hypothesis.
    """
    input_texts: List[str] = []
    options_str = create_options(options)
    for context, hypothesis in zip(contexts, hypotheses):
        input_texts.append(
            # TEMPLATE_NLI.format(
            #     context=context, hypothesis=hypothesis, options=options_str
            # )
            TEMPLATE_NLI_3.format(
                context=context, hypothesis=hypothesis, options=options_str
            )
            # TEMPLATE_BOOL_Q.format(
            #     context=context, question=hypothesis, options=options_str
            # )
        )
    output_texts = gen_texts(model, tokenizer, input_texts, max_length=max_length)
    # for q, a in zip(input_texts, output_texts):
    #     print("------")
    #     print(q)
    #     print(a)
    #     exit()
    return output_texts


def gen_texts_bool_qa(
    model, tokenizer, statements: List[str], max_length: int = 20
) -> List[str]:
    """
    Ask the model whether a statement is true or not.
    The prompt is "Is it true that {statement}?\n\n- yes\n- no"
    """
    input_texts: List[str] = []
    options = "OPTIONS:\n- yes\n- no"
    for statement in statements:
        statement = statement[:-1]  # -1 to remove period.
        input_texts.append(
            TEMPLATE_BOOL_QA.format(statement=statement, options=options)
        )
    output_texts = gen_texts(model, tokenizer, input_texts, max_length=max_length)
    # for q, a in zip(input_texts, output_texts):
    #     print("------")
    #     print(q)
    #     print(a)
    #     exit()
    return output_texts
