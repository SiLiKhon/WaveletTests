from typing import Tuple, Union, Optional
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pywt


class TabulatedFunc:
    def __init__(
        self,
        xx: Optional[np.array] = None,
        yy: Optional[np.array] = None,
        outval: float = 0.0
    ):
        if yy is None:
            yy = np.array([0.0, 0.0])

        if xx is None:
            xx = np.linspace(0.0, 1.0, len(yy))

        assert xx.ndim == 1
        assert xx.shape == yy.shape
        assert (xx[1:] - xx[:-1] > 0).all()
        self.xx = xx
        self.yy = yy
        self.outval = outval

    def shift(self, delta) -> "TabulatedFunc":
        return TabulatedFunc(self.xx + delta, self.yy)

    def scale(self, factor, keep_int: bool = True) -> "TabulatedFunc":
        renorm_factor = 1
        if keep_int:
            renorm_factor /= factor**0.5

        return TabulatedFunc(
            self.xx * factor,
            self.yy * renorm_factor,
            outval=self.outval * renorm_factor,
        )

    def der(self) -> "TabulatedFunc":
        return TabulatedFunc(
            (self.xx[1:] + self.xx[:-1]) / 2,
            (self.yy[1:] - self.yy[:-1]) / (self.xx[1:] - self.xx[:-1])
        )

    def int(self) -> "TabulatedFunc":
        return TabulatedFunc(
            (self.xx[1:] + self.xx[:-1]) / 2,
            ((self.yy[1:] + self.yy[:-1]) * (self.xx[1:] - self.xx[:-1]) / 2).cumsum(),
        )

    def resample(self, xx) -> "TabulatedFunc":
        return TabulatedFunc(xx, np.interp(xx, self.xx, self.yy, left=self.outval, right=self.outval))

    def _binary_op(self, other: Union["TabulatedFunc", float], op: str) -> "TabulatedFunc":
        if not isinstance(other, TabulatedFunc):
            other = TabulatedFunc(
                np.array([self.xx[0], self.xx[-1]]),
                np.array([other, other]),
                outval=other,
            )
    
        xx = np.unique(np.concatenate([self.xx, other.xx]))
        yy_this = self.resample(xx).yy
        yy_other = other.resample(xx).yy
        return TabulatedFunc(
            xx,
            eval(f"yy_this {op} yy_other"),
            outval=eval(f"self.outval {op} other.outval")
        )

    def __mul__(self, other: Union["TabulatedFunc", float]) -> "TabulatedFunc":
        return self._binary_op(other, "*")

    def __add__(self, other: Union["TabulatedFunc", float]) -> "TabulatedFunc":
        return self._binary_op(other, "+")

    def __rmul__(self, other: float) -> "TabulatedFunc":
        return self * other

    def __radd__(self, other: float) -> "TabulatedFunc":
        return self + other

    def plot(self, **kwargs):
        return plt.plot(self.xx, self.yy, **kwargs)

@lru_cache
def get_phi_psi(wvt, level: int) -> Tuple[TabulatedFunc, TabulatedFunc]:
    phi, psi, xx = wvt.wavefun(level)
    return TabulatedFunc(xx, phi), TabulatedFunc(xx - (xx[0] + xx[-1] - 1) / 2, psi)

class MyWavelet:
    def __init__(self, wvt: pywt.Wavelet):
        self.wvt = wvt

    def get_phi_psi(self, level: int) -> Tuple[TabulatedFunc, TabulatedFunc]:
        return get_phi_psi(self.wvt, level)

    # def first_der(self, level: int) -> Tuple[np.array, np.array, np.array]:
    #     phi, psi, xx = self.wvt.wavefun(level)

    #     h = (xx[-1] - xx[0]) / (len(xx) - 1)
    #     assert np.isclose(h, xx[1] - xx[0])
    #     phi_ = (phi[1:] - phi[:-1]) / h
    #     psi_ = (psi[1:] - psi[:-1]) / h
    #     return phi_, psi_, (xx[1:] + xx[:-1]) / 2

    # def second_der(self, level: int) -> Tuple[np.array, np.array, np.array]:
    #     phi_, psi_, xx = self.first_der(level)
    #     return (phi_[1:] - phi_[:-1]) / h, (psi_[1:] - psi_[:-1]) / h, xx[1: -1]

# phi, psi = MyWavelet(pywt.Wavelet("sym8")).get_phi_psi(5)

# phi.plot()
# (phi * 2 + 4).plot()
# (1 + (-3) * phi).plot()
# # phi.der().int().plot()
# # phi.int().der().plot()
# # # phi_ = phi.der()
# # # phi_.plot()
# # # phi_.der().plot()
# # # plt.figure()
# # # psi.plot()
# # # psi_ = psi.der()
# # # psi_.plot()
# # # psi_.der().plot()

# # # print(phi.int()[-1], psi.int()[-1])
# print((phi * psi).int().yy[-1])


if __name__ == "__main__":
    wvt = pywt.Wavelet("sym8")

    xx = np.linspace(0, 1, 2**18)
    yy = -xx**2 * (1 - xx**1) * (0.4 - xx) * np.exp(-((xx - 0.5) * 10)**2)
    src_func = TabulatedFunc(xx, yy)
    src_func.plot(zorder=10);
    # plt.plot(xx, yy, zorder=10);

    coefs = pywt.wavedec(yy, wvt, mode='smooth')
    for c in coefs:
        print(c.shape)

    phi, _ = MyWavelet(wvt).get_phi_psi(13)
    # plt.figure()
    # phi.plot()
    result_func = (
        sum(
            phi.shift(i) * c * 2**(-7) for i, c in enumerate(coefs[0])
        ).shift(-14).scale(1 / 16, False)
    )
    result_func.plot(linewidth=3)

    (result_func * TabulatedFunc(yy=np.array([1.0, 1.0])) + (-1) * src_func).plot()

    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.03, 0.03)
    plt.grid()
    print(xx.shape, phi.xx.shape)
    plt.show()
