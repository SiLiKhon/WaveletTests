from typing import List, Optional, Collection, Union

import numpy as np
import sympy as sp
from scipy.special import binom
import pywt


def calculate_moments(wvt: pywt.Wavelet, num_moments: Optional[int] = None) -> np.array:
    hh = np.array(wvt.filter_bank[2])
    if num_moments is None:
        num_moments = len(hh) // 2
    moments = np.empty(num_moments, dtype=np.float64)
    moments[0] = 1.0

    M = (np.arange(len(hh))[:, None]**np.arange(num_moments, 0, -1)[None, :] * hh[:, None]).sum(axis=0)
    for p in range(1, num_moments):
        bin_coef = binom(p, np.arange(p))
        moments[p] = (M[-p:] * bin_coef * moments[:p]).sum() / np.sqrt(2) / (2**p - 1)
    return moments

def lagrange_polynomials(x: sp.Symbol, roots: Collection[Union[int, float, sp.Symbol]]) -> List[sp.Expr]:
    numerator_terms = [x - root for root in roots]
    polynomials = []
    for i, root_i in enumerate(roots):
        terms = sp.prod(numerator_terms[:i] + numerator_terms[i+1:])
        polynomials.append(
            terms / terms.subs({x: root_i})
        )

    return polynomials

def get_Plr(roots: Collection[Union[int, float, sp.Symbol]]) -> sp.Matrix:
    x = sp.Symbol("x")
    return sp.Matrix(
        [sp.Poly(p, x).all_coeffs()[::-1] for p in lagrange_polynomials(x, roots)]
    )

def calculate_quad_filter(wvt: pywt.Wavelet) -> np.array:
    moments = calculate_moments(wvt, len(wvt.filter_bank[2]))
    plr = get_Plr(range(len(moments)))
    return np.array([float(x) for x in plr @ moments])
