import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    positions = np.arange(seq_len)[:, np.newaxis]
    even_dims = np.arange(0, d_model,2)

    div_terms = base ** (even_dims / d_model)
    angles = positions / div_terms

    pe = np.zeros((seq_len, d_model))

    pe[:, 0::2] = np.sin(angles)

    print(angles)

    num_odd_cols = pe[:, 1::2].shape[1]
    print(num_odd_cols)
    pe[:, 1::2] = np.cos(angles[:, :num_odd_cols])

    return pe


print (positional_encoding(6,7))
