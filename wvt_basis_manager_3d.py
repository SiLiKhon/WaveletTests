from typing import Union, Tuple, Callable

import numpy as np
from wvt_utils import calculate_quad_filter, calculate_diff_filter

import pywt

class WaveletBasisManager3d:
    def __init__(
        self,
        wvt: pywt.Wavelet,
        nsteps: Union[int, Tuple[int, int, int]],
        lims_x: Tuple[float, float] = (0.0, 1.0),
        lims_y: Tuple[float, float] = (0.0, 1.0),
        lims_z: Tuple[float, float] = (0.0, 1.0),
    ):
        assert lims_x[0] < lims_x[1]
        assert lims_y[0] < lims_y[1]
        assert lims_z[0] < lims_z[1]

        if isinstance(nsteps, int):
            nsteps = (nsteps, nsteps, nsteps)
        self.nsteps = np.array(nsteps)

        self.lims_x = lims_x
        self.lims_y = lims_y
        self.lims_z = lims_z
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
        self._knots_z = np.linspace(*self.lims_z, num_knots[2])
        self._scale_factor_x = (self.lims_x[1] - self.lims_x[0]) / (num_knots[0] + 1)
        self._scale_factor_y = (self.lims_y[1] - self.lims_y[0]) / (num_knots[1] + 1)
        self._scale_factor_z = (self.lims_z[1] - self.lims_z[0]) / (num_knots[2] + 1)

    def _iterate_overlapping_ids(self, axis: str):
        assert len(axis) == 1
        basis_size = self.nsteps["xyz".index(axis)]

        for i in range(basis_size):
            for delta_i in range(2 * self.m - 1):
                j = i + delta_i
                if j < basis_size:
                    yield (i, j)

    def get_ke_matrix(self) -> np.array:
        matrices = []
        for axis in "xyz":
            basis_size = self.nsteps["xyz".index(axis)]
            matrices.append(np.zeros((basis_size, basis_size)))
            scale_factor = getattr(self, f"scale_factor_{axis}")
            for i, j in self._iterate_overlapping_ids(axis):
                matrices[-1][i, j] = matrices[-1][j, i] = (
                    -self.aa[2 * self.m - 2 + j - i] / 2 / scale_factor**2
                )

        return (
            matrices[0][:, None, None, :, None, None]
            + matrices[1][None, :, None, None, :, None]
            + matrices[2][None, None, :, None, None, :]
        )

    def get_matrix_elements(self, func: Callable) -> np.array:
        knots_x, knots_y, knots_z = np.meshgrid(
            self.knots_x, self.knots_y, self.knots_z, indexing="ij"
        )
        f_knots = func(knots_x, knots_y, knots_z)
        mat = np.zeros((
            self.nsteps[0],
            self.nsteps[1],
            self.nsteps[2],
            self.nsteps[0],
            self.nsteps[1],
            self.nsteps[2],
        ))

        for ix, jx in self._iterate_overlapping_ids("x"):
            for iy, jy in self._iterate_overlapping_ids("y"):
                for iz, jz in self._iterate_overlapping_ids("z"):
                    ll_x = np.arange(jx, ix + 2 * self.m)[:, None, None]
                    ll_y = np.arange(jy, iy + 2 * self.m)[None, :, None]
                    ll_z = np.arange(jz, iz + 2 * self.m)[None, None, :]
                    value = (
                        f_knots[ll_x, ll_y, ll_z]
                        * self.ww[ll_x - ix] * self.ww[ll_x - jx]
                        * self.ww[ll_y - iy] * self.ww[ll_y - jy]
                        * self.ww[ll_z - iz] * self.ww[ll_z - jz]
                    ).sum()
                    mat[ix, iy, iz, jx, jy, jz] = value
                    mat[ix, iy, jz, jx, jy, iz] = value
                    mat[ix, jy, iz, jx, iy, jz] = value
                    mat[ix, jy, jz, jx, iy, iz] = value
                    mat[jx, iy, iz, ix, jy, jz] = value
                    mat[jx, iy, jz, ix, jy, iz] = value
                    mat[jx, jy, iz, ix, iy, jz] = value
                    mat[jx, jy, jz, ix, iy, iz] = value

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
    def knots_z(self) -> np.array:
        return self._knots_z

    @property
    def scale_factor_x(self) -> np.array:
        return self._scale_factor_x

    @property
    def scale_factor_y(self) -> np.array:
        return self._scale_factor_y

    @property
    def scale_factor_z(self) -> np.array:
        return self._scale_factor_z
