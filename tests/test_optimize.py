import numpy as np
from ivim.models import NO_REGIME, DIFFUSIVE_REGIME, BALLISTIC_REGIME, SBALLISTIC_REGIME
from ivim.seq.sde import MONOPOLAR, BIPOLAR
from ivim.optimize import crlb

D = np.array([0.5e-3, 1e-3, 1.5e-3])
f = np.array([0.05, 0.15, 0.05])
K = np.array([1.0, 1.2, 2.0])
Dstar = np.array([10e-3, 15e-3, 20e-3])
vd = np.array([1.5, 2.0, 2.0])*0.1
bthr = 200
usr_input = {'Gmax': 60e-3, 't_epi': 30e-3, 't_180':1.3e-3,'t_rise':0.7e-3}
SNR = 100

# Test functions
def test_crlb_sIVIM():
    for fitK in [True, False]:
        bmax = 1500
        for minbias in [True, False]:
            if minbias:
                bthr = 1
                bias_regimes = [DIFFUSIVE_REGIME, BALLISTIC_REGIME]
            else:
                bthr = 200
                bias_regimes = [DIFFUSIVE_REGIME]
            for bias_regime in bias_regimes:
                b, a, _ = crlb(D, f, NO_REGIME, bmax = bmax, fitK = fitK, minbias = minbias, bias_regime = bias_regime, K = K, SNR = SNR, 
                        bthr = bthr, Dstar = Dstar, vd = vd,seq=MONOPOLAR,usr_input=usr_input)
                
                np.testing.assert_equal(b[0], 0) # First b-value is by design = 0
                np.testing.assert_almost_equal(np.sum(a), 1.0, 2) # a should sum to one
                np.testing.assert_equal(b.size, 3+fitK)
                np.testing.assert_equal(a.size, 3+fitK)
                np.testing.assert_array_less(b, bmax*(1+1e-5))
                np.testing.assert_array_less(bthr, b[1:]*(1+1e-5))
                np.testing.assert_array_less(a, 1.0)
                np.testing.assert_array_less(0.0, a)
                
def test_crlb_diffusive():
    bmax = 1000
    b, a, _ = crlb(D, f, DIFFUSIVE_REGIME, bmax = bmax, fitK = False, minbias = False, 
                bias_regime = DIFFUSIVE_REGIME, K = K, SNR = SNR, Dstar = Dstar,usr_input=usr_input)
    np.testing.assert_almost_equal(np.sum(a), 1.0, 2) # a should sum to one
    np.testing.assert_equal(b.size, 4)
    np.testing.assert_equal(a.size, 4)
    np.testing.assert_array_less(b, bmax*(1+1e-5))
    np.testing.assert_array_less(0, b + 1e-5)
    np.testing.assert_array_less(a, 1.0)
    np.testing.assert_array_less(0.0, a)

def test_crlb_ballistic():
    for seq in [MONOPOLAR, BIPOLAR]:
        if seq == MONOPOLAR:
            bmax = 1000
        else:
            bmax = 400
        b, a, c, _ = crlb(D, f, BALLISTIC_REGIME, bmax = bmax, fitK = False, minbias = False, bias_regime = BALLISTIC_REGIME, K = K, SNR = SNR,
                    Dstar = Dstar, vd = vd, seq=seq, usr_input=usr_input)
        np.testing.assert_almost_equal(np.sum(a), 1.0, 2) # a should sum to one
        np.testing.assert_equal(b.size, 4)
        np.testing.assert_equal(a.size, 4)
        np.testing.assert_equal(c.size, 4)
        np.testing.assert_array_less(b, bmax*(1+1e-5))
        np.testing.assert_array_less(0, b + 1e-5)
        np.testing.assert_array_less(a, 1.0)
        np.testing.assert_array_less(0.0, a)
        np.testing.assert_array_less(0, c + 1e-5)

def test_crlb_sballistic():
    for seq in [MONOPOLAR, BIPOLAR]:
        if seq == MONOPOLAR:
            bmax = 1000
        else:
            bmax = 400
        b, a, c, _ = crlb(D, f, SBALLISTIC_REGIME, bmax = bmax, fitK = False, minbias = False, bias_regime = BALLISTIC_REGIME, K = K, SNR = SNR,
                        seq=seq, usr_input=usr_input)
        np.testing.assert_almost_equal(np.sum(a), 1.0, 2) # a should sum to one
        np.testing.assert_equal(b.size, 3)
        np.testing.assert_equal(a.size, 3)
        np.testing.assert_equal(c.size, 3)
        np.testing.assert_array_less(b, bmax*(1+1e-5))
        np.testing.assert_array_less(0, b + 1e-5)
        np.testing.assert_array_less(a, 1.0)
        np.testing.assert_array_less(0.0, a)
        np.testing.assert_array_less(0, c + 1e-5)