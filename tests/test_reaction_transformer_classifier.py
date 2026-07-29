from pathlib import Path

import pytest
import torch

from chempilot.reactions.tokenization import (
    ReactionSmilesTokenizer,
)
from chempilot.reactions.transformer_classifier import (
    ReactionTransformerClassifierConfig,
    ReactionTransformerMultiLabelClassifier,
)


CHECKPOINT = Path(
    "artifacts/pretrained/day5/"
    "rxnfp_bert_pretrained"
)


def test_invalid_number_of_labels():
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        ReactionTransformerClassifierConfig(
            number_of_labels=0
        )


def test_invalid_unfreeze_count():
    with pytest.raises(
        ValueError,
        match="between 0 and 12",
    ):
        ReactionTransformerClassifierConfig(
            number_of_labels=20,
            unfreeze_last_n_layers=13,
        )


@pytest.fixture(scope="module")
def model_contract():
    if not CHECKPOINT.exists():
        pytest.skip(
            "RXNFP checkpoint not prepared"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = ReactionSmilesTokenizer(
        CHECKPOINT / "vocab.txt"
    )

    model = (
        ReactionTransformerMultiLabelClassifier(
            ReactionTransformerClassifierConfig(
                number_of_labels=20,
                checkpoint_directory=(
                    CHECKPOINT
                ),
                pooling="masked_mean",
                unfreeze_last_n_layers=2,
                dropout=0.1,
            )
        )
        .to(device)
    )

    return model, tokenizer, device


def make_batch(
    tokenizer,
    device,
):
    encoded = tokenizer.tokenize_reactions(
        [
            "CCO>>CC=O",
            "CC(=O)O>>CC(=O)N",
        ],
        max_length=256,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    return {
        name: tensor.to(device)
        for name, tensor in (
            encoded.items()
        )
    }


def test_trainable_layers(
    model_contract,
):
    model, _, _ = model_contract

    assert (
        model
        .trainable_encoder_layer_indices
        == [10, 11]
    )

    assert not (
        model.encoder.embeddings
        .word_embeddings.weight
        .requires_grad
    )

    assert (
        model.classifier.weight
        .requires_grad
    )


def test_parameter_summary(
    model_contract,
):
    model, _, _ = model_contract

    summary = model.parameter_summary()

    assert summary[
        "encoder_total"
    ] == 6_674_432

    assert summary[
        "encoder_trainable"
    ] == 1_054_208

    assert summary[
        "classifier_trainable"
    ] == 5_140

    assert summary[
        "trainable_encoder_layers"
    ] == [10, 11]


def test_forward_shape_and_finiteness(
    model_contract,
):
    model, tokenizer, device = (
        model_contract
    )

    model.eval()

    batch = make_batch(
        tokenizer,
        device,
    )

    with torch.inference_mode():
        logits = model(**batch)

    assert logits.shape == (
        2,
        20,
    )

    assert torch.isfinite(
        logits
    ).all()


def test_gradient_scope(
    model_contract,
):
    model, tokenizer, device = (
        model_contract
    )

    model.train()
    model.zero_grad(set_to_none=True)

    batch = make_batch(
        tokenizer,
        device,
    )

    logits = model(**batch)
    loss = logits.square().mean()
    loss.backward()

    assert (
        model.classifier.weight.grad
        is not None
    )

    assert (
        model.encoder.encoder.layer[
            11
        ].attention.self.query.weight.grad
        is not None
    )

    assert (
        model.encoder.encoder.layer[
            0
        ].attention.self.query.weight.grad
        is None
    )

    assert (
        model.encoder.embeddings
        .word_embeddings.weight.grad
        is None
    )


def test_eval_is_deterministic(
    model_contract,
):
    model, tokenizer, device = (
        model_contract
    )

    model.eval()

    batch = make_batch(
        tokenizer,
        device,
    )

    with torch.inference_mode():
        first = model(**batch)
        second = model(**batch)

    assert torch.equal(
        first,
        second,
    )