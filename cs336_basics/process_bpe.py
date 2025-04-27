from collections import Counter

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(path: str,
              vocab_size: int,
              special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    text = read_text(path)

    return train_bpe_v2(text, vocab_size, special_tokens)


def train_bpe_v2(text: str,
                 vocab_size: int,
                 special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    chunks, vocab = init_bpe(text, special_tokens)

    merges = list()
    pre_tokens_counter = Counter()

    for chunk in chunks:
        pre_tokens = [m.group(0) for m in re.finditer(PAT, chunk)]
        pre_tokens_counter.update(pre_tokens)

    pre_token_bytes = {}
    for token in pre_tokens_counter:
        pre_token_bytes[token] = token.encode("utf-8")

    token_pairs = Counter()
    for token, value in pre_tokens_counter.items():
        bytes_tokens = pre_token_bytes[token]
        for a, b in zip(bytes_tokens, bytes_tokens[1:]):
            token_pairs[(a, b)] += value

    while len(vocab) < vocab_size and token_pairs:
        max_value = max(token_pairs.values())

        candidates = [p for p, cnt in token_pairs.items() if cnt == max_value]
        max_pair = max(
            candidates,
            key=lambda pair: (vocab[pair[0]], vocab[pair[1]])
        )

        a_id, b_id = max_pair
        merges.append((vocab[a_id], vocab[b_id]))

        new_bytes = vocab[a_id] + vocab[b_id]
        new_id = len(vocab)
        vocab[new_id] = new_bytes

        for token, byte_seq in pre_token_bytes.items():
            i = 0
            new_seq = list()
            while i < len(byte_seq):
                if i + 1 < len(byte_seq) and (byte_seq[i], byte_seq[i + 1]) == max_pair:
                    new_seq.append(new_id)
                    i += 2
                else:
                    new_seq.append(byte_seq[i])
                    i += 1
            pre_token_bytes[token] = new_seq

        token_pairs = Counter()
        for token, value in pre_tokens_counter.items():
            bytes_tokens = pre_token_bytes[token]
            for a, b in zip(bytes_tokens, bytes_tokens[1:]):
                token_pairs[(a, b)] += value

    print(merges)
    return vocab, merges


def init_bpe(text: str, special_tokens: list[str]) -> tuple[list[str], dict[int, bytes]]:
    vocab = {i: bytes([i]) for i in range(256)}

    # add from specila tokens
    chunks = re.split("|".join(map(re.escape, special_tokens)), text)

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    return chunks, vocab


def read_text(path: str) -> str:
    with path.open("r", encoding="utf-8") as f:
        text = f.read()
    return text


def train_bpe_ai(path: str, vocab_size: int, special_tokens: list[str]):
    """
    Train a byte-level BPE tokenizer on the text at input_path.

    Args:
        input_path (str): Path to the training text file.
        vocab_size (int): Maximum vocabulary size (including initial bytes and special tokens).
        special_tokens (list[str]): List of special tokens to add to the vocabulary.

    Returns:
        vocab (dict[int, bytes]): Mapping from token ID to token bytes.
        merges (list[tuple[bytes, bytes]]): Ordered list of byte-pair merges.
    """
    # Regex pre-tokenizer pattern (GPT-2 style)
    # pat = re.compile(PAT)

    # Initialize vocabulary with all single bytes
    vocab = {i: bytes([i]) for i in range(256)}
    # Add special tokens (if not already present)
    for tok in special_tokens or []:
        tok_bytes = tok.encode('utf-8')
        if tok_bytes not in vocab.values():
            vocab[len(vocab)] = tok_bytes

    # Read full corpus
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read()

    # Split on special tokens to avoid merges across document boundaries
    if special_tokens:
        escaped = [re.escape(tok) for tok in special_tokens]
        split_pat = re.compile('|'.join(escaped))
        chunks = split_pat.split(data)
    else:
        chunks = [data]

    # Pre-tokenize and count token frequencies
    token_counts = Counter()
    for chunk in chunks:
        for m in re.finditer(PAT, chunk):
            token = m.group(0).encode('utf-8')
            token_counts[token] += 1

    # Represent each pre-token as a sequence of byte tokens (tuple of bytes)
    seq_counts = {}
    for token_bytes, cnt in token_counts.items():
        # Break token_bytes into sequence of single-byte tokens
        seq = tuple(bytes([b]) for b in token_bytes)
        seq_counts[seq] = cnt

    merges = []
    # Iteratively merge until vocab size reached
    while len(vocab) < vocab_size:
        # Count all adjacent byte-pair frequencies
        pair_freqs = Counter()
        for seq, cnt in seq_counts.items():
            for a, b in zip(seq, seq[1:]):
                pair_freqs[(a, b)] += cnt
        if not pair_freqs:
            break
        # Find most frequent pair; break ties by lex order
        max_freq = max(pair_freqs.values())
        candidates = [pair for pair, freq in pair_freqs.items() if freq == max_freq]
        best = max(candidates)
        merges.append(best)
        merged = best[0] + best[1]
        # Add to vocab
        vocab[len(vocab)] = merged

        # Update sequences by replacing occurrences of best pair
        new_seq_counts = {}
        for seq, cnt in seq_counts.items():
            new_seq = []
            i = 0
            while i < len(seq):
                # If matching pair, merge
                if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best:
                    new_seq.append(merged)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_seq_counts[tuple(new_seq)] = cnt
        seq_counts = new_seq_counts

    return vocab, merges


if __name__ == "__main__":
    str = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
<|endoftext|>
iron cement is a ready for use paste which is laid as a fillet by putty knife or finger in the mould edges ( corners ) of the steel ingot mould .
iron cement protects the ingot against the hot , abrasive steel casting process .
a fire restant repair cement for fire places , ovens , open fireplaces etc .
    """.strip()
    vocab, merges = train_bpe_v2(
        text=str,
        vocab_size=300,
        special_tokens=["<|endoftext|>"]
    )
    print(vocab)
    print(merges)
