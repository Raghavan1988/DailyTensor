from collections import Counter

def word_count_dict(sentences):
    return dict(Counter(word for sentence in sentences for word in sentence))
