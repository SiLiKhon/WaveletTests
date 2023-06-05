import matplotlib.pyplot as plt
import pywt
import numpy as np

from tabulated_wvt import TabulatedFunc, TabulatedWavelet

if __name__ == "__main__":
    wvt = pywt.Wavelet("sym8")

    xx = np.linspace(0, 1, 2**18)
    yy = -xx**2 * (1 - xx**1) * (0.4 - xx) * np.exp(-((xx - 0.5) * 10)**2)
    src_func = TabulatedFunc(xx, yy)
    src_func.plot(zorder=10)

    coefs = pywt.wavedec(yy, wvt, mode='smooth')
    print("--- Wavedec coefficients shapes ---")
    for c in coefs:
        print(c.shape)
    print("-----------------------------------")

    phi, _ = TabulatedWavelet(wvt).get_phi_psi(13)
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
    print("Num signal samples:", len(xx))
    print("Num wvt samples:", len(phi.xx))
    plt.show()
