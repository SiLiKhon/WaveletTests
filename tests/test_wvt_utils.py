from typing import Callable
from pytest import fixture
from pywt import Wavelet
import numpy as np

from tabulated_wvt import TabulatedWavelet, TabulatedFunc
from wvt_utils import calculate_moments, calculate_quad_filter


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
def phi(wavelet: TabulatedWavelet) -> TabulatedFunc:
    f, _ = wavelet.get_phi_psi(level=15)
    return f

@fixture
def moments_numeric(moments_analytical: np.array, phi: TabulatedFunc) -> np.array:
    num_moments = len(moments_analytical)
    return np.array([
        (phi * TabulatedFunc(phi.xx, phi.xx**p)).int().yy[-1]
        for p in range(num_moments)
    ])

def test_moments(moments_analytical: np.array, moments_numeric: np.array) -> None:
    np.testing.assert_allclose(moments_numeric, moments_analytical, atol=0, rtol=1e-3)

@fixture
def polynomial() -> Callable:
    return lambda xx: (
        0.5 * xx**4
        -1.32 * xx**3
        -5.87 * xx**2
        + 57.1 * xx
        - 15.402
    )

@fixture
def projection_coefficient_numeric(polynomial: Callable, phi: TabulatedFunc) -> float:
    return (phi * TabulatedFunc(phi.xx, polynomial(phi.xx))).int().yy[-1]

@fixture
def quad_filter(wavelet: TabulatedWavelet) -> np.array:
    return calculate_quad_filter(wavelet.wvt)

@fixture
def projection_coefficient_quad(polynomial: Callable, quad_filter: np.array) -> float:
    return (polynomial(np.arange(len(quad_filter))) * quad_filter).sum()

def test_quad_filter(projection_coefficient_numeric: float, projection_coefficient_quad: float):
    np.testing.assert_allclose(projection_coefficient_numeric, projection_coefficient_quad, atol=0, rtol=1e-3)
