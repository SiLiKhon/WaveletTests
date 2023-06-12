from typing import Tuple, List, Callable

import numpy as np

from tabulated_wvt import TabulatedWavelet, TabulatedFunc
from wvt_utils import calculate_quad_filter, calculate_diff_filter

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
    def base_wvt(self) -> TabulatedWavelet:
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

        self._knots = np.linspace(
            *self.lims, np.round((self.lims[1] - self.lims[0]) / self.scale_factor).astype(int) + 1,
        )
        xxmin_vals = np.array([f.xx.min() for f in self.basis_funcs])
        assert np.allclose(xxmin_vals, self.knots[:len(self.basis_funcs)])

        self._ww = calculate_quad_filter(wvt.wvt)
        self._aa = calculate_diff_filter(wvt.wvt)

    def reconstruct(self, coefs: np.array):
        assert coefs.ndim == 1
        assert len(coefs) == len(self.basis_funcs)
        return sum(
            func * c for c, func in zip(coefs, self.basis_funcs)
        )

    def get_matrix_elements(self, func: Callable) -> np.array:
        if self._basis_funcs_psi:
            raise NotImplementedError("Quadratures not implemented for subscale yet")

        basis_size = len(self.basis_funcs)
        mat = np.zeros((basis_size, basis_size))
        f_knots = func(self.knots)
        for i, j in self.iterate_overlapping_ids():
            ll = np.arange(j, i + 2 * self.m)
            mat[i, j] = mat[j, i] = (f_knots[ll] * self.ww[ll - i] * self.ww[ll - j]).sum()

        return mat

    def get_ke_matrix(self) -> np.array:
        basis_size = len(self.basis_funcs)
        mat = np.zeros((basis_size, basis_size))
        for i, j in self.iterate_overlapping_ids():
            mat[i, j] = mat[j, i] = -self.aa[2 * self.m - 2 + j - i] / 2 / self.scale_factor**2
        return mat

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
    def m(self) -> int:
        return self._m

    @property
    def basis_funcs(self) -> List[TabulatedFunc]:
        return self._basis_funcs

    @property
    def base_phi(self) -> TabulatedFunc:
        return self._base_phi

    @property
    def base_psi(self) -> TabulatedFunc:
        return self._base_psi

    @property
    def scale_factor(self) -> float:
        return self._scale_factor

    @property
    def knots(self) -> np.array:
        return self._knots

    @property
    def ww(self) -> np.array:
        return self._ww

    @property
    def aa(self) -> np.array:
        return self._aa
