from typing import Tuple, List

import numpy as np

from tabulated_wvt import TabulatedWavelet, TabulatedFunc

class WaveletBasisManager:
    def __init__(
        self,
        wvt: TabulatedWavelet,
        nsteps: int,
        lims: Tuple[float, float] = (0.0, 1.0),
        level: int = 5,
    ):
        assert lims[0] < lims[1]
        assert nsteps > 0
        self.level = level
        self.nsteps = nsteps
        self.lims = lims
        self.base_wvt = wvt

    @property
    def base_wvt(self):
        return self._base_wvt

    @base_wvt.setter
    def base_wvt(self, wvt: TabulatedWavelet):
        _, _, xx = wvt.wvt.wavefun(1)
        assert xx[0] == 0
        m = (xx[-1] + 1) / 2
        assert int(m) == round(m)

        self._base_wvt = wvt
        self._m = int(m)
        self._base_phi, self._base_psi = wvt.get_phi_psi(self.level)
        self._scale_factor = (self.lims[1] - self.lims[0]) / (self.nsteps + 2 * (self.m - 1))

        self._base_phi = self.base_phi.scale(self.scale_factor)
        self._base_psi = self.base_psi.scale(self.scale_factor)

        self._basis_funcs = [
            self.base_phi.shift(self.lims[0] + i * self.scale_factor)
            for i in range(self.nsteps)
        ]

    def reconstruct(self, coefs: np.array):
        assert coefs.ndim == 1
        assert len(coefs) == len(self.basis_funcs)
        return sum(
            func * c for c, func in zip(coefs, self.basis_funcs)
        )

    def iterate_overlapping_ids(self):
        for i1 in range(len(self.basis_funcs)):
            for i2 in range(i1, min(i1 + 2 * self.m - 1, len(self.basis_funcs))):
                yield (i1, i2)

    @property
    def m(self):
        return self._m

    @property
    def basis_funcs(self) -> List[TabulatedFunc]:
        return self._basis_funcs

    @property
    def base_phi(self):
        return self._base_phi

    @property
    def base_psi(self):
        return self._base_psi

    @property
    def scale_factor(self):
        return self._scale_factor
