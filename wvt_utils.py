from typing import List, Optional

import numpy as np
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
