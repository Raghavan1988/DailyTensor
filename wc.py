from collections import Counter


def word_count_dict(sentences, lowercase=False):
    """
    Count word occurrences across a collection of sentences.

    Each sentence may be either a pre-tokenized list of words or a raw
    string (which is split on whitespace). Set lowercase=True to make the
    count case-insensitive.

    >>> word_count_dict([["a", "b"], "a c"])
    {'a': 2, 'b': 1, 'c': 1}
    """
    def tokens(sentence):
        words = sentence.split() if isinstance(sentence, str) else sentence
        return (w.lower() for w in words) if lowercase else words

    return dict(Counter(word for sentence in sentences for word in tokens(sentence)))
