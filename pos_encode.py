import numpy as np


def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Sinusoidal positional encoding of shape (seq_len, d_model).

        PE(pos, 2i)   = sin(pos / base^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / base^(2i / d_model))

    Works for odd d_model too: the last (even) column gets a sin term with
    no matching cos partner.
    """
    positions = np.arange(seq_len)[:, np.newaxis]
    even_dims = np.arange(0, d_model, 2)

    div_terms = base ** (even_dims / d_model)
    angles = positions / div_terms

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles)

    # With odd d_model there is one fewer cos column than sin column.
    num_odd_cols = pe[:, 1::2].shape[1]
    pe[:, 1::2] = np.cos(angles[:, :num_odd_cols])

    return pe


if __name__ == "__main__":
    print(positional_encoding(6, 7))
