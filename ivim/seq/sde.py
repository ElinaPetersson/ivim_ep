""" Functions related to basic dMRI pulse sequences with trapezoidal gradient pulses. """

import os
import numpy as np
import numpy.typing as npt
from ivim.constants import y as gamma
from ivim.io.base import read_bval, write_cval, read_cval

# String contants
MONOPOLAR = 'monopolar'
BIPOLAR   = 'bipolar'

# Functions
def calc_b(G: npt.NDArray[np.float64], Delta: float, delta: float, seq: str = MONOPOLAR) -> npt.NDArray[np.float64]:
    """
    Calculate b-value given other relevant pulse sequence parameters.

    Arguments:
        G:     gradient strength      [T/mm] (Note the units preferred to get b-values in commonly used unit)
        Delta: gradient separation    [s]
        delta: gradient duration      [s]
        seq:   (optional) pulse sequence (monopolar or bipolar)

    Output:
        b:     b-value [s/mm2]
    """

    b = gamma**2 * G**2 * delta**2 * (Delta-delta/3)
    if seq == BIPOLAR:
        b *= 2
    elif seq != MONOPOLAR:
        raise ValueError(f'Unknown pulse sequence: "{seq}"')
    return b

def calc_c(G: npt.NDArray[np.float64], Delta: float, delta:float, seq: str = MONOPOLAR, fc: bool = False) -> npt.NDArray[np.float64]:
    """
    Calculate c-value (flow encoding) given other relevant pulse sequence parameters.

    Arguments:
        G:     gradient strength      [T/mm] (Note the units preferred to get b-values in commonly used units)
        Delta: gradient separation    [s]
        delta: gradient duration      [s]
        seq:   (optional) pulse sequence (monopolar or bipolar)
        fc:    (optional) specify is the pulse sequence is flow compensated (only possible for bipolar)

    Output:
        c:     c-value [s/mm]
    """    
    
    c = gamma * G * delta * Delta
    if seq == BIPOLAR:
        if fc:
            c = np.zeros_like(G)
        else:
            c *= 2
    elif seq == MONOPOLAR:
        if fc:
            raise ValueError(f'monopolar pulse sequence cannot be flow compensated.')
    else:
        raise ValueError(f'Unknown pulse sequence: "{seq}". Valid options are "{MONOPOLAR}" and "{BIPOLAR}".')
    return c

def G_from_b(b: npt.NDArray[np.float64], Delta: float, delta: float, seq: str = MONOPOLAR) -> npt.NDArray[np.float64]:
    """
    Calculate gradient strength given other relevant pulse sequence parameters.

    Arguments:
        b:     b-value                [s/mm2]
        Delta: gradient separation    [s]
        delta: gradient duration      [s]
        seq:   (optional) pulse sequence (monopolar or bipolar)

    Output:
        G:     gradient strength      [T/mm]
    """    

    G = np.sqrt(b / calc_b(np.ones_like(b), Delta, delta, seq))
    if (np.isnan(G)).any():
        if isinstance(G,np.ndarray):
            G[np.where(np.isnan(G))] = 0
        else:
            G = 0
    return G

def cval_from_bval(bval_file: str, Delta: float, delta: float, seq: str = MONOPOLAR, cval_file: str = '', fc: bool = False) -> npt.NDArray[np.float64]:
    """
    Write .cval based on .bval file and other relevant pulse sequence parameters.

    Arguments:
        bval_file: path to .bval file
        Delta:     gradient separation
        delta:     gradient duration
        seq:       (optional) pulse sequence (monopolar or bipolar)
        cval_file: (optional) path to .cval file. Will use the .bval path if not set
    """

    b = read_bval(bval_file)
    c = calc_c(G_from_b(b, Delta, delta, seq), Delta, delta, seq, fc)
    if cval_file == '':
        cval_file = os.path.splitext(bval_file)[0] + '.cval'
    write_cval(cval_file,c)

def calc_interm_pars(b: npt.NDArray[np.float64], usr_input: dict, seq = MONOPOLAR, nfc: int = 0):
    """
    Calculate parameters for the intermediate regime given other relevant pulse sequence parameters.

    Arguments:
        b:          b-value                 [s/mm2]
        usr_input:  dict with the gradient rise time, maximum gradient strength and duration of the 180 pulse [s]
        seq:        (optional) pulse sequence (monopolar or bipolar)
        nfc:        (optional) number of flow compensated gradients, required for biopolar sequence
   
    Output:
        Delta:      gradient separation     [s]
        delta:      gradient duration       [s]
        c:          c-values                [s/mm]
        k:          +/-1 for non-compensated/flow-compensated
        T:          encoding time           [s]
    """   
    if seq == BIPOLAR:      
        r = np.roots([4/3, 2*usr_input['t_rise'],0,-max(b)*1e6/(gamma**2*usr_input['Gmax']**2)])
        delta = r[(r.real>=0)*(r.imag == 0)][0].real
        Delta = delta + usr_input['t_rise']
        c = np.zeros_like(b)
        c[nfc:] = calc_c(G_from_b(b[nfc:], Delta, delta, BIPOLAR), Delta, delta, BIPOLAR)
        k=np.array([int(ci!=0)-int(ci==0) for ci in c])
        T = np.ones_like(b)*(Delta*2+delta*2+usr_input['t_180']+usr_input['t_rise'])
    elif seq == MONOPOLAR:
        r = np.roots([2/3, usr_input['t_180'],0,-max(b)*1e6/(gamma**2*usr_input['Gmax']**2)])
        delta = r[(r.real>=0)*(r.imag == 0)][0].real
        Delta = delta + usr_input['t_180']
        c = calc_c(G_from_b(b, Delta, delta, MONOPOLAR), Delta, delta, MONOPOLAR)
        k = np.ones_like(b)
        T = np.ones_like(b)*(Delta+delta)
    return Delta*np.ones_like(b), delta*np.ones_like(b), c, k, T