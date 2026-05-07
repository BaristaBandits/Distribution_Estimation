# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cpdef double compute_normalizer_cython(
    object model,
    object history,
    int k_syn,
    double beta
):
    cdef double first_sum = 0.0
    cdef double total
    cdef double lam

    cdef object w
    cdef object s
    cdef object synonyms
    cdef object candidate_words

    # =====================================
    # Cached lambda
    # =====================================
    lam = model.semantic_lambda_cache[history]

    # =====================================
    # Build sparse candidate set
    # =====================================
    candidate_words = set()

    # Base history followers
    candidate_words.update(
        model.followers_of_history.get(
            history,
            []
        )
    )

    # Synonym followers
    synonyms = model.get_synonyms(
        history,
        k_syn
    )

    for s, _ in synonyms:

        candidate_words.update(
            model.followers_of_history.get(
                s,
                []
            )
        )

    # =====================================
    # Sparse first-term sum
    # =====================================
    for w in candidate_words:

        first_sum += model.semantic_first_term(
            history,
            w,
            k_syn,
            beta
        )

    # =====================================
    # Continuation contribution
    # =====================================
    total = (
        first_sum
        + lam * model.global_semantic_cont_sum
    )

    if total <= 0:
        return 1.0

    return total
