import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    Implements self attention
    """

    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.attention = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)  # [batch_size, seq_len, embed_size]
        K = self.key(x)
        V = self.value(x)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(Q.size(-1), dtype=torch.float32)
        )
        attention_weights = self.attention(scores)  # [batch_size, seq_len, seq_len]

        # Apply attention to values
        out = torch.matmul(attention_weights, V)  # [batch_size, seq_len, embed_size]
        return out


class TCNN(nn.Module):
    """
    TCNN Architecture

    Consists of two convolutional layers followed by batch normalization, and activation function
    Then two fully connected layers follow, the last one is for the classification

    Attention can be applied or not, between the convolutional and the fully connected layers
    """

    def __init__(
        self,
        num_features,
        out_channels,
        kernel_size,
        sequence_length=10,
        dropout=0.3,
        attention=False,
    ):
        super(TCNN, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=1,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels * 2,
            kernel_size=kernel_size,
            padding=1,
        )

        self.bn2 = nn.BatchNorm1d(out_channels * 2)

        self.fc1 = nn.Linear(
            out_channels * 2 * (sequence_length - 2 * (kernel_size - 3)),
            out_channels * 4,
        )
        self.fc2 = nn.Linear(out_channels * 4, 2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if attention:
            self.attention = SelfAttention(embed_size=out_channels * 2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        if hasattr(self, "attention"):
            x = x.permute(0, 2, 1)
            x = self.attention(x)
            x = x.permute(0, 2, 1)

        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        logits = self.fc2(x)

        return logits
