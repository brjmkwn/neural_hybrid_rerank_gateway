"""Mathematical and geometric utilities for vector operations."""
import numpy as np


def l2_normalize(vectors: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:


    """
    Scales vectors so their Euclidean length (L2 norm) is exactly 1.0.

    Args:
        vectors: NumPy array of shape (d,) or (N, d).
        axis: Axis along which to compute the norm (default is the feature dimension, -1).
        eps: Small epsilon guard to prevent division by zero for zero vectors.

    Returns:
        Normalized array of the same shape with unit length vectors.
    """
    # Compute the norm along the specified axis, keeping dimensions for broadcasting
    norm = np.linalg.norm(vectors, axis=axis, keepdims=True)
    
    # Numerical stability guard against division by zero
    norm = np.maximum(norm, eps)
    
    return vectors / norm


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
    normalized: bool = False,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Computes cosine similarity between vector/matrix 'a' and vector/matrix 'b'.

    If normalized=True, assumes inputs are already unit L2-normalized and computes
    a fast dot product (matrix multiplication).

    Args:
        a: Query vector (d,) or batch matrix (N, d).
        b: Document vector (d,) or corpus matrix (M, d).
        normalized: Whether inputs are already L2-normalized (length = 1.0).
        eps: Epsilon guard for unnormalized inputs.

    Returns:
        NumPy array of cosine similarities bounded in [-1.0, 1.0].
    """
    if not normalized:
        a = l2_normalize(a, axis=-1, eps=eps)
        b = l2_normalize(b, axis=-1, eps=eps)

    # If both are 1D vectors: dot product -> scalar float
    if a.ndim == 1 and b.ndim == 1:
        sim = np.dot(a, b)
    # If a is 1D (d,) and b is 2D (M, d): vector-matrix product -> (M,)
    elif a.ndim == 1 and b.ndim == 2:
        sim = np.dot(b, a)
    # If a is 2D (N, d) and b is 2D (M, d): matrix product -> (N, M)
    elif a.ndim == 2 and b.ndim == 2:
        sim = np.dot(a, b.T)
    # If a is 2D (N, d) and b is 1D (d,): matrix-vector product -> (N,)
    elif a.ndim == 2 and b.ndim == 1:
        sim = np.dot(a, b)
    else:
        raise ValueError(f"Unsupported array dimensions: a={a.ndim}D, b={b.ndim}D")

    # Guard against precision overshoot (e.g., 1.0000000000000002)
    return np.clip(sim, -1.0, 1.0)


def compute_token_frequencies(tokens: list[str]) -> dict[str, int]:
    """
    Computes raw term frequencies across a sequence of tokens.
    Returns a dictionary mapping token -> count, ordered by frequency descending.
    """
    from collections import Counter
    counts = Counter(tokens)
    return dict(counts.most_common())


def compute_term_frequencies(
    tokens: list[str],
    normalize: bool = True,
) -> dict[str, float]:
    """
    Computes term frequencies (TF) for a document token list.

    Args:
        tokens: List of string tokens in a document.
        normalize: If True, divides raw count by total document length (relative TF).
                   If False, returns raw integer counts as floats.

    Returns:
        Dictionary mapping token -> term frequency (float).
    """
    if not tokens:
        return {}

    raw_counts = compute_token_frequencies(tokens)
    total_tokens = len(tokens)

    if not normalize:
        return {term: float(cnt) for term, cnt in raw_counts.items()}

    return {term: cnt / total_tokens for term, cnt in raw_counts.items()}


def compute_shannon_idf(total_docs: int, doc_freq: int) -> float:
    """
    Computes classic Shannon Information / IDF weight for a term.
    IDF = ln(total_docs / doc_freq)

    Args:
        total_docs: Total number of documents in the corpus (N).
        doc_freq: Number of documents containing the term (n).

    Returns:
        Shannon information content in nats (natural log).
    """
    if total_docs <= 0 or doc_freq <= 0:
        return 0.0
    return float(np.log(total_docs / doc_freq))


def compute_bm25_smoothed_idf(total_docs: int, doc_freq: int) -> float:
    """
    Computes the Robertson-Spärck Jones smoothed IDF used in modern BM25Okapi:
    IDF = ln(1 + (N - n + 0.5) / (n + 0.5))

    Guarantees:
    - Never produces negative weights even when n > N / 2.
    - Gracefully handles doc_freq = 0 and total_docs = 0.

    Args:
        total_docs: Total number of documents in the corpus (N).
        doc_freq: Number of documents containing the term (n).

    Returns:
        Non-negative smoothed BM25 IDF weight.
    """
    if total_docs <= 0 or doc_freq <= 0:
        return 0.0

    # RSJ smoothed formulation: ln(1 + (N - n + 0.5) / (n + 0.5))
    fraction = (total_docs - doc_freq + 0.5) / (doc_freq + 0.5)
    return float(np.log(1.0 + max(0.0, fraction)))





