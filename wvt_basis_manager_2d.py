from typing import Union, Tuple, Callable

import numpy as np
from wvt_utils import calculate_quad_filter, calculate_diff_filter

import pywt

class WaveletBasisManager2d:
    def __init__(
        self,
        wvt: pywt.Wavelet,
        nsteps: Union[int, Tuple[int, int]],
        lims_x: Tuple[float, float] = (0.0, 1.0),
        lims_y: Tuple[float, float] = (0.0, 1.0),
    ):
        assert lims_x[0] < lims_x[1]
        assert lims_y[0] < lims_y[1]

        if isinstance(nsteps, int):
            nsteps = (nsteps, nsteps)
        self.nsteps = np.array(nsteps)

        self.lims_x = lims_x
        self.lims_y = lims_y
        self.base_wvt = wvt

    @property
    def base_wvt(self) -> pywt.Wavelet:
        return self._base_wvt

    @base_wvt.setter
    def base_wvt(self, wvt: pywt.Wavelet):
        _, _, xx = wvt.wavefun(1)
        assert xx[0] == 0
        m = (xx[-1] + 1) / 2
        assert int(m) == round(m)

        self._base_wvt = wvt
        self._m = int(m)
        self._ww = calculate_quad_filter(wvt)
        self._aa = calculate_diff_filter(wvt)
        num_knots = (self.nsteps + 2 * self.m - 1)
        self._knots_x = np.linspace(*self.lims_x, num_knots[0])
        self._knots_y = np.linspace(*self.lims_y, num_knots[1])
        self._scale_factor_x = (self.lims_x[1] - self.lims_x[0]) / (num_knots[0] - 1)
        self._scale_factor_y = (self.lims_y[1] - self.lims_y[0]) / (num_knots[1] - 1)

    def reconstruct(self, coefs: np.array, level: int = 3):
        if coefs.ndim == 1:
            coefs = coefs.reshape(*self.nsteps)

        assert coefs.shape == tuple(self.nsteps)
        phi_func, _, phi_xx = self.base_wvt.wavefun(level=level)
        assert phi_xx[0] == 0
        assert phi_xx[-1] == 2 * self.m - 1
        full_size = (2 * self.m - 2 + self.nsteps) * (len(phi_xx) - 1) / (2 * self.m - 1) + 1
        assert (full_size.astype(int) == full_size).all()
        full_size = full_size.astype(int)
        knot_step_size = (len(phi_xx) - 1) / (2 * self.m - 1)
        assert int(knot_step_size) == knot_step_size
        knot_step_size = int(knot_step_size)
        assert phi_xx[knot_step_size] == 1

        xx = np.linspace(*self.lims_x, full_size[0])
        yy = np.linspace(*self.lims_x, full_size[1])

        ff = np.zeros(shape=tuple(full_size), dtype=np.float64)
        phi_2d = phi_func[:, None] * phi_func[None, :]
        for ix, row_i in enumerate(coefs):
            for iy, c in enumerate(row_i):
                ff[
                    ix * knot_step_size: (ix + 2 * self.m - 1) * knot_step_size + 1,
                    iy * knot_step_size: (iy + 2 * self.m - 1) * knot_step_size + 1,
                ] += phi_2d * c

        return xx, yy, ff

    def _iterate_overlapping_ids(self, axis: str):
        assert len(axis) == 1
        basis_size = self.nsteps["xy".index(axis)]

        for i in range(basis_size):
            for delta_i in range(2 * self.m - 1):
                j = i + delta_i
                if j < basis_size:
                    yield (i, j)

    def get_ke_matrix(self) -> np.array:
        matrices = []
        for axis in "xy":
            basis_size = self.nsteps["xy".index(axis)]
            matrices.append(np.zeros((basis_size, basis_size)))
            scale_factor = getattr(self, f"scale_factor_{axis}")
            for i, j in self._iterate_overlapping_ids(axis):
                matrices[-1][i, j] = matrices[-1][j, i] = (
                    -self.aa[2 * self.m - 2 + j - i] / 2 / scale_factor**2
                )

        return (
            matrices[0][:, None, :, None] * np.eye(self.nsteps[1])[None, :, None, :]
            + matrices[1][None, :, None, :] * np.eye(self.nsteps[0])[:, None, :, None]
        )

    def get_matrix_elements(self, func: Callable) -> np.array:
        knots_x, knots_y = np.meshgrid(
            self.knots_x, self.knots_y, indexing="ij"
        )
        f_knots = func(knots_x, knots_y)
        mat = np.zeros((
            self.nsteps[0],
            self.nsteps[1],
            self.nsteps[0],
            self.nsteps[1],
        ))

        for ix, jx in self._iterate_overlapping_ids("x"):
            for iy, jy in self._iterate_overlapping_ids("y"):
                ll_x = np.arange(jx, ix + 2 * self.m)[:, None]
                ll_y = np.arange(jy, iy + 2 * self.m)[None, :]
                value = (
                    f_knots[ll_x, ll_y]
                    * self.ww[ll_x - ix] * self.ww[ll_x - jx]
                    * self.ww[ll_y - iy] * self.ww[ll_y - jy]
                ).sum()
                mat[ix, iy, jx, jy] = value
                mat[ix, jy, jx, iy] = value
                mat[jx, iy, ix, jy] = value
                mat[jx, jy, ix, iy] = value

        return mat

    @property
    def m(self) -> int:
        return self._m

    @property
    def ww(self) -> np.array:
        return self._ww

    @property
    def aa(self) -> np.array:
        return self._aa

    @property
    def knots_x(self) -> np.array:
        return self._knots_x

    @property
    def knots_y(self) -> np.array:
        return self._knots_y

    @property
    def scale_factor_x(self) -> np.array:
        return self._scale_factor_x

    @property
    def scale_factor_y(self) -> np.array:
        return self._scale_factor_y

