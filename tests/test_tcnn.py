import pytest
import torch
import torch.nn as nn
from torch import nn
from apple_stock_extremes_ml.models.tcnn import SelfAttention, TCNN


# Test SelfAttention
def test_self_attention_initialization():
    embed_size = 8
    attention_layer = SelfAttention(embed_size)

    assert isinstance(attention_layer, SelfAttention)
    assert attention_layer.query.weight.size() == (embed_size, embed_size)
    assert attention_layer.key.weight.size() == (embed_size, embed_size)
    assert attention_layer.value.weight.size() == (embed_size, embed_size)


def test_self_attention_forward():
    embed_size = 8
    attention_layer = SelfAttention(embed_size)

    x = torch.randn(4, 10, embed_size)
    out = attention_layer(x)

    assert out.size() == x.size()


# Test TCNN
@pytest.mark.parametrize("attention", [True, False])
def test_tcnn_initialization(attention):
    num_features = 6  # e.g., 6 features from stock data (Open, Volume, etc.)
    out_channels = 16
    kernel_size = 3
    sequence_length = 10

    tcnn_model = TCNN(
        num_features=num_features,
        out_channels=out_channels,
        kernel_size=kernel_size,
        sequence_length=sequence_length,
        attention=attention,
    )

    assert isinstance(tcnn_model, TCNN)
    assert isinstance(tcnn_model.conv1, nn.Conv1d)
    assert isinstance(tcnn_model.conv2, nn.Conv1d)
    assert isinstance(tcnn_model.fc1, nn.Linear)
    assert isinstance(tcnn_model.fc2, nn.Linear)

    if attention:
        assert hasattr(tcnn_model, "attention")
        assert isinstance(tcnn_model.attention, SelfAttention)
    else:
        assert not hasattr(tcnn_model, "attention")


def test_tcnn_forward():
    num_features = 6  # e.g., 6 features from stock data (Open, Volume, etc.)
    out_channels = 16
    kernel_size = 3
    sequence_length = 10

    # Test with attention
    tcnn_model_with_attention = TCNN(
        num_features=num_features,
        out_channels=out_channels,
        kernel_size=kernel_size,
        sequence_length=sequence_length,
        attention=True,
    )

    # Test without attention
    tcnn_model_without_attention = TCNN(
        num_features=num_features,
        out_channels=out_channels,
        kernel_size=kernel_size,
        sequence_length=sequence_length,
        attention=False,
    )

    x = torch.randn(4, num_features, sequence_length)

    # Forward pass with attention
    out_with_attention = tcnn_model_with_attention(x)
    assert out_with_attention.size(0) == 4
    assert out_with_attention.size(1) == 2

    # Forward pass without attention
    out_without_attention = tcnn_model_without_attention(x)
    assert out_without_attention.size(0) == 4
    assert out_without_attention.size(1) == 2
