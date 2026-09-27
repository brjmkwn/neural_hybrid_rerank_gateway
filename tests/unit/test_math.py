"""Unit tests for mathematical utilities."""
import numpy as np
from gateway.math_utils import (
    compute_bm25_smoothed_idf,
    compute_shannon_idf,
    compute_term_frequencies,
    compute_token_frequencies,
    cosine_similarity,
    l2_normalize,
)









def test_l2_normalize_single_vector() -> None:
    """Test L2 normalization on a single 1D vector."""
    vec = np.array([3.0, 4.0])
    norm_vec = l2_normalize(vec)

    # 3-4-5 triangle: length is 5, normalized elements are 3/5=0.6 and 4/5=0.8
    assert np.isclose(np.linalg.norm(norm_vec), 1.0)
    assert np.allclose(norm_vec, np.array([0.6, 0.8]))


def test_l2_normalize_batch_matrix() -> None:
    """Test L2 normalization across a 2D batch of vectors."""
    batch = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    norm_batch = l2_normalize(batch)

    # Check that the shape is preserved (3, 3)
    assert norm_batch.shape == (3, 3)

    # Each row vector must have unit norm
    row_norms = np.linalg.norm(norm_batch, axis=1)
    assert np.allclose(row_norms, np.array([1.0, 1.0, 1.0]))


def test_l2_normalize_zero_vector_no_crash() -> None:
    """Test numerical stability: all-zero vector must not yield NaN or crash."""
    zero_vec = np.array([0.0, 0.0, 0.0])
    norm_vec = l2_normalize(zero_vec)

    assert not np.isnan(norm_vec).any()
    assert np.all(norm_vec <= 1e-10)


def test_cosine_similarity_cardinal_directions() -> None:
    """Test identical (1.0), opposite (-1.0), and orthogonal (0.0) vectors."""
    v1 = np.array([1.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v_opp = np.array([-1.0, 0.0])
    v_orth = np.array([0.0, 1.0])

    # Identical vectors -> 1.0
    assert np.isclose(cosine_similarity(v1, v2), 1.0)
    # Opposite vectors -> -1.0
    assert np.isclose(cosine_similarity(v1, v_opp), -1.0)
    # Orthogonal vectors -> 0.0
    assert np.isclose(cosine_similarity(v1, v_orth), 0.0)


def test_cosine_similarity_matrix_against_vector() -> None:
    """Test 1 query vector against a matrix of 3 documents."""
    query = np.array([1.0, 0.0, 0.0])
    docs = np.array([
        [1.0, 0.0, 0.0],   # identical -> 1.0
        [0.0, 1.0, 0.0],   # orthogonal -> 0.0
        [-1.0, 0.0, 0.0],  # opposite -> -1.0
    ])

    scores = cosine_similarity(query, docs)
    assert scores.shape == (3,)
    assert np.allclose(scores, np.array([1.0, 0.0, -1.0]))


def test_cosine_similarity_normalized_flag_speed_equivalence() -> None:
    """Verify that normalized=True matches normalized=False on pre-normalized data."""
    rng = np.random.default_rng(42)
    raw_q = rng.standard_normal((10, 64))
    raw_docs = rng.standard_normal((50, 64))

    score_unnorm = cosine_similarity(raw_q, raw_docs, normalized=False)

    norm_q = l2_normalize(raw_q)
    norm_docs = l2_normalize(raw_docs)
    score_norm = cosine_similarity(norm_q, norm_docs, normalized=True)

    assert np.allclose(score_unnorm, score_norm)


def test_dot_product_monotonicity_with_angle() -> None:
    """
    Step 005: Verify that cosine similarity / dot product decreases monotonically
    as the angle between vectors increases from 0 to 180 degrees.
    """
    # Fixed reference vector at angle 0 (along the X axis)
    ref_vec = np.array([1.0, 0.0])

    # Generate test vectors rotated at angles from 0 to 180 degrees (in radians)
    angles_deg = np.linspace(0.0, 180.0, num=19)  # 0, 10, 20, ..., 180 deg
    angles_rad = np.radians(angles_deg)

    test_vectors = np.column_stack([np.cos(angles_rad), np.sin(angles_rad)])

    # Compute similarity between reference vector and all rotated vectors
    scores = cosine_similarity(ref_vec, test_vectors, normalized=True)

    # Monotonicity check: each score must be strictly <= previous score
    diffs = np.diff(scores)
    assert np.all(diffs <= 1e-7), f"Monotonicity violated! Differences: {diffs}"

    # Verify boundary values
    assert np.isclose(scores[0], 1.0)    # 0 degrees -> 1.0
    assert np.isclose(scores[9], 0.0)    # 90 degrees -> 0.0
    assert np.isclose(scores[-1], -1.0)  # 180 degrees -> -1.0


def test_compute_token_frequencies_zipf_ordering() -> None:
    """Step 007: Test token frequency calculation and descending popularity rank."""
    tokens = ["the", "gateway", "the", "rerank", "the", "gateway", "neural"]
    counts = compute_token_frequencies(tokens)

    # 'the' appears 3 times, 'gateway' 2 times, 'rerank' 1 time, 'neural' 1 time
    keys = list(counts.keys())
    assert keys[0] == "the"
    assert keys[1] == "gateway"
    assert counts["the"] == 3
    assert counts["gateway"] == 2
    assert counts["rerank"] == 1


def test_compute_term_frequencies_normalized_and_raw() -> None:
    """Step 008: Test TF calculations (both raw counts and relative length normalization)."""
    # 4 tokens total: 'neural' (2), 'gateway' (1), 'fast' (1)
    tokens = ["neural", "gateway", "neural", "fast"]

    raw_tf = compute_term_frequencies(tokens, normalize=False)
    assert raw_tf["neural"] == 2.0
    assert raw_tf["gateway"] == 1.0

    rel_tf = compute_term_frequencies(tokens, normalize=True)
    assert np.isclose(rel_tf["neural"], 2.0 / 4.0)   # 0.50
    assert np.isclose(rel_tf["gateway"], 1.0 / 4.0)  # 0.25
    assert np.isclose(rel_tf["fast"], 1.0 / 4.0)     # 0.25

    # Empty document edge case
    assert compute_term_frequencies([], normalize=True) == {}


def test_compute_shannon_idf() -> None:
    """Step 009: Test Shannon information content / raw IDF."""
    # Ubiquitous token (appears in 100/100 docs) -> 0.0 nats
    assert np.isclose(compute_shannon_idf(total_docs=100, doc_freq=100), 0.0)

    # Rare token (appears in 1 out of 100 docs) -> ln(100) ≈ 4.605 nats
    assert np.isclose(compute_shannon_idf(total_docs=100, doc_freq=1), np.log(100))

    # Edge cases
    assert compute_shannon_idf(0, 0) == 0.0
    assert compute_shannon_idf(10, 0) == 0.0


def test_smoothed_idf_non_negativity_proof() -> None:
    """
    Steps 010 & 011: Mathematical proof that smoothed IDF never produces negative weights,
    even when a token appears in 50%, 90%, 99%, or 100% of all documents.
    """
    total_docs = 1000

    # Test doc frequencies from n=1 all the way to n=1000 (100% of docs)
    for n in range(1, total_docs + 1):
        idf = compute_bm25_smoothed_idf(total_docs=total_docs, doc_freq=n)
        # Invariant: smoothed IDF must strictly be >= 0.0
        assert idf >= 0.0, f"Negative IDF produced at n={n}: {idf}"

    # Rare token (n=1 in N=1000) should have high positive IDF weight
    rare_idf = compute_bm25_smoothed_idf(total_docs=1000, doc_freq=1)
    assert rare_idf > 6.0

    # Ubiquitous token (n=1000 in N=1000) should have near-zero but non-negative weight
    ubiquitous_idf = compute_bm25_smoothed_idf(total_docs=1000, doc_freq=1000)
    assert ubiquitous_idf >= 0.0








