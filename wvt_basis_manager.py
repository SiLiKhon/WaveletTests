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
        use_subscale: bool = False,
    ):
        assert lims[0] < lims[1]
        assert nsteps > 0
        self.use_subscale = use_subscale
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
        self._base_psi = self._base_psi.shift(-self._base_psi.xx[0])
        self._scale_factor = (self.lims[1] - self.lims[0]) / (self.nsteps + 2 * (self.m - 1))

        self._base_phi = self.base_phi.scale(self.scale_factor)
        self._base_psi = self.base_psi.scale(self.scale_factor)

        self._basis_funcs_phi = [
            self.base_phi.shift(self.lims[0] + i * self.scale_factor)
            for i in range(self.nsteps)
        ]
        self._basis_funcs_psi = [
            self.base_psi.shift(self.lims[0] + i * self.scale_factor)
            for i in range(self.nsteps)
        ] if self.use_subscale else []
        self._basis_funcs = self._basis_funcs_phi + self._basis_funcs_psi

    def reconstruct(self, coefs: np.array):
        assert coefs.ndim == 1
        assert len(coefs) == len(self.basis_funcs)
        return sum(
            func * c for c, func in zip(coefs, self.basis_funcs)
        )

    def iterate_overlapping_ids(self):
        for i in range(len(self._basis_funcs_phi)):
            for delta_i in range(2 * self.m - 1):
                j = i + delta_i
                if j < len(self._basis_funcs_phi):
                    yield (i, j)

                if self._basis_funcs_psi:
                    if j < len(self._basis_funcs_phi):
                        yield (i, len(self._basis_funcs_phi) + j)
                        yield (len(self._basis_funcs_phi) + i, len(self._basis_funcs_phi) + j)

                    if delta_i > 0:
                        j -= 2 * delta_i
                        if j >= 0:
                            yield (i, len(self._basis_funcs_phi) + j)

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
