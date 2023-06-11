# Scaling function moments formula not so hard to derive.
# Also present in https://doi.org/10.1063/1.478741 (though with a typo in normalization).
#
# Quadrature filter uses coefficients of the Lagrange polynomials
# (https://doi.org/10.1016/j.jcp.2006.01.003)
#
# Kinetic energy filter (diff_filter) taken from
# https://comphys.unibas.ch/publications/Goedecker1998d.pdf
# (Stefan Goedecker, Wavelets and their application for the solution of partial differential
# equations in physics, chapter 23)

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

def build_Aij(hh: Collection[float]) -> np.array:
    hh = np.array(hh)
    mu = np.arange(len(hh))[:, None, None, None]
    nu = np.arange(len(hh))[None, :, None, None]
    ii = np.arange(-len(hh) + 2, len(hh) - 1)[None, None, :, None]
    jj = np.arange(-len(hh) + 2, len(hh) - 1)[None, None, None, :]

    return (
        hh[mu] * hh[nu] * (jj == 2 * ii - nu + mu).astype(int)
    ).sum(axis=(0, 1))

def get_diff_filter(wvt: pywt.Wavelet) -> np.array:
    hh = wvt.filter_bank[2]
    Aij = build_Aij(hh)
    eigvals, eigvecs = np.linalg.eig(Aij)
    i_value = np.abs(eigvals - 0.25).argmin()
    assert np.isclose(eigvals[i_value], 0.25)
    assert (np.abs(np.imag(eigvecs[i_value])) < 1e-8).all()

    aa = np.real(eigvecs[:, i_value])
    norm = (np.arange(-len(hh) + 2, len(hh) - 1)**2 * aa).sum() / 2
    return aa / norm
