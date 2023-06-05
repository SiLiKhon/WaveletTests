import matplotlib.pyplot as plt
import pywt
import numpy as np

from tabulated_wvt import TabulatedFunc, TabulatedWavelet

if __name__ == "__main__":
    wvt = pywt.Wavelet("sym8")

    SCALE = 16 * 2
    NSTEPS = 10 * 2
    SHIFT = 0 * 2

    xx = np.linspace(0, 1, 2**10)
    yy = -xx**2 * (1 - xx**1) * (0.4 - xx) * np.exp(-((xx - 0.5) * 10)**2)
    src_func = TabulatedFunc(xx, yy)

    phi, _ = TabulatedWavelet(wvt).get_phi_psi(5)
    phi_k = phi.scale(1 / SCALE)
    coefs = [(phi_k.shift((SHIFT + i) / SCALE) * src_func).int().yy[-1] for i in range(NSTEPS)]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.stem(coefs)
    plt.title("coefficients")

    plt.subplot(1, 2, 2)
    src_func.plot(zorder=10, label="source func")
    result_func = (sum(phi_k.shift((SHIFT + i) / SCALE) * c for i, c in enumerate(coefs)))
    result_func.plot(linewidth=3, label="reconstructed func")

    (result_func * TabulatedFunc(yy=np.array([1.0, 1.0])) + (-1) * src_func).plot(label="difference")
    plt.legend()

    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.03, 0.03)
    plt.grid()
    print("Num signal samples:", len(xx))
    print("Num wvt samples:", len(phi.xx))
    plt.show()
