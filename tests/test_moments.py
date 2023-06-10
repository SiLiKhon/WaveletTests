from pytest import fixture
from pywt import Wavelet
import numpy as np

from tabulated_wvt import TabulatedWavelet, TabulatedFunc
from wvt_utils import calculate_moments


@fixture(params=[
    "db4", "db5", "db6", "db7", "db8",
    "sym4", "sym5", "sym6", "sym7", "sym8",
    "coif4",
])
def wvt_name(request) -> str:
    return request.param

@fixture
def wavelet(wvt_name: str) -> TabulatedWavelet:
    return TabulatedWavelet(Wavelet(wvt_name))

@fixture
def moments_analytical(wavelet: TabulatedWavelet) -> np.array:
    return calculate_moments(wavelet.wvt)

@fixture
def moments_numeric(wavelet: TabulatedWavelet) -> np.array:
    phi, _ = wavelet.get_phi_psi(level=15)
    num_moments = len(wavelet.wvt.filter_bank[2]) // 2
    return np.array([
        (phi * TabulatedFunc(phi.xx, phi.xx**p)).int().yy[-1]
        for p in range(num_moments)
    ])

def test_moments(moments_analytical: np.array, moments_numeric: np.array) -> None:
    np.testing.assert_allclose(moments_numeric, moments_analytical, atol=0, rtol=1e-3)
