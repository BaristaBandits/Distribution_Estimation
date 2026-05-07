# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cpdef double compute_normalizer_cython(
    object model,
    object history,
    int k_syn,
    double beta
):
    cdef double total = 0.0
    cdef object w
    cdef double first
    cdef double lam
    cdef double p_cont
    cdef double p

    lam = model.semantic_lambda_cache[history]

    for w in model.vocab:

        first = model.semantic_first_term(
            history,
            w,
            k_syn,
            beta
        )

        p_cont = model.semantic_cont_cache[w]

        p = first + lam * p_cont

        if p > 0:
            total += p

    if total <= 0:
        return 1.0

    return total
