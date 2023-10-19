"""Unit tests for construction of temperature ladders"""

import numpy as np

import pytest

import DTMCMC.temperature_ladder_helpers as th

TEST_DATA_DIR = 'tests/test_data/'

#set of parameters to use for several tests
test_set1 = [
        (1,2,1.),(1,3,1.),(2,4,1.),(2,32,1.),(1,32,1.),(1,32,0.9),(2,32,0.9),(1,32,1.1),(2,32,1.1),(1,32,9.9),(2,32,9.9),(8,8,1.),(7,8,1.),(6,8,1.),(5,8,1.),(4,8,1.),(3,8,1.),(2,8,1.),(1,8,1.),(1,1,1.),(1,1,np.inf),(8,8,np.inf),(4,8,np.inf),(1,8,np.inf),(2,1,1.),(4,3,1.),(4,2,1.)
]

@pytest.mark.parametrize("n_cold,n_chain,T_cold",test_set1)
def test_entropy_spacing_fromfile_inf(n_cold,n_chain,T_cold):
    """test the entropy based spacing produces results that makes sense"""
    if n_cold > n_chain:
        with pytest.raises(ValueError):
            T_ladder = th.entropy_ladder_fromfile(n_chain,n_cold,TEST_DATA_DIR+'gal1_Ts_resample.npy',TEST_DATA_DIR+'gal1_logL_var_resample.npy',use_inf_final=True,T_cold=T_cold)

        return

    if n_cold == n_chain:
        with pytest.warns(UserWarning):
            T_ladder = th.entropy_ladder_fromfile(n_chain,n_cold,TEST_DATA_DIR+'gal1_Ts_resample.npy',TEST_DATA_DIR+'gal1_logL_var_resample.npy',use_inf_final=True,T_cold=T_cold)
            Ts_in = T_ladder.Ts
    else:
        T_ladder = th.entropy_ladder_fromfile(n_chain,n_cold,TEST_DATA_DIR+'gal1_Ts_resample.npy',TEST_DATA_DIR+'gal1_logL_var_resample.npy',use_inf_final=True,T_cold=T_cold)
        Ts_in = T_ladder.Ts

    # Strict technical requirements
    assert Ts_in.size == n_chain                # check correct number of chains
    assert np.sum(Ts_in==T_cold) >= n_cold      # check correct number of cold chains
    assert not np.any(Ts_in < 0.)               # check no negative temperature chains
    assert np.all(T_ladder.Ts == Ts_in)         # check object matches
    assert np.all(T_ladder.betas[np.isfinite(Ts_in)&(Ts_in>0)]==1./Ts_in[np.isfinite(Ts_in)&(Ts_in>0)]) # check inverses match
    assert np.all(Ts_in[T_ladder.betas==0.] == np.inf) # check inverses match
    assert np.all(T_ladder.betas[Ts_in==0.] == np.inf) # check inverses match


    # Not technically required, but expected in this test case
    if not T_cold == np.inf:
        assert np.sum(Ts_in==T_cold) == n_cold                   # check correct number of cold chains
        if n_chain > n_cold:
            assert np.sum(Ts_in == np.inf) == 1                      # check 1 infinite temperature chain exists
        assert np.all(np.diff(Ts_in[n_cold:]) >= 0.)             # check non-cold chains are sorted
    else:
        assert np.sum(Ts_in==T_cold) == min(n_cold+1,n_chain)

    assert np.unique(Ts_in[n_cold:]).size == n_chain-n_cold  # check all temperatures are unique
    if T_cold <= 1.:
        assert np.all(Ts_in[n_cold:] > T_cold)                   # check all higher temperatures warmer than cold chain

@pytest.mark.parametrize("n_cold,n_chain,T_cold",test_set1)
def test_entropy_spacing_fromfile_noinf(n_cold,n_chain,T_cold):
    """test the entropy based spacing produces results that makes sense"""
    if n_cold > n_chain:
        with pytest.raises(ValueError):
            T_ladder = th.entropy_ladder_fromfile(n_chain,n_cold,TEST_DATA_DIR+'gal1_Ts_resample.npy',TEST_DATA_DIR+'gal1_logL_var_resample.npy',use_inf_final=False,T_cold=T_cold)

        return

    T_ladder = th.entropy_ladder_fromfile(n_chain,n_cold,TEST_DATA_DIR+'gal1_Ts_resample.npy',TEST_DATA_DIR+'gal1_logL_var_resample.npy',use_inf_final=False,T_cold=T_cold)
    Ts_in = T_ladder.Ts

    # Strict technical requirements
    assert Ts_in.size == n_chain                # check correct number of chains
    assert np.sum(Ts_in==T_cold) >= n_cold      # check correct number of cold chains
    assert not np.any(Ts_in < 0.)               # check no negative temperature chains
    assert np.all(T_ladder.Ts == Ts_in)         # check object matches
    assert np.all(T_ladder.betas[np.isfinite(Ts_in)&(Ts_in>0)]==1./Ts_in[np.isfinite(Ts_in)&(Ts_in>0)]) # check inverses match
    assert np.all(Ts_in[T_ladder.betas==0.] == np.inf) # check inverses match
    assert np.all(T_ladder.betas[Ts_in==0.] == np.inf) # check inverses match


    # Not technically required, but expected in this test case
    if not T_cold == np.inf:
        assert np.sum(Ts_in==T_cold) == n_cold                   # check correct number of cold chains
        if n_chain > n_cold:
            assert np.sum(Ts_in == np.inf) <= 1                  # check not inserting more non infinite chains (note this method does not *guarantee* temps are finite))
        assert np.all(np.diff(Ts_in[n_cold:]) >= 0.)             # check non-cold chains are sorted
    else:
        assert np.sum(Ts_in==T_cold) == min(n_cold+1,n_chain)

    assert np.unique(Ts_in[n_cold:]).size == n_chain-n_cold  # check all temperatures are unique
    if T_cold <= 1.:
        assert np.all(Ts_in[n_cold:] > T_cold)                   # check all higher temperatures warmer than cold chain


@pytest.mark.parametrize("n_cold,n_chain,T_cold",test_set1)
def test_geometric_spacing_inf(n_cold,n_chain,T_cold):
    """test the entropy based spacing produces results that makes sense"""
    T_min = 1.
    T_max = 1000.

    if n_cold > n_chain:
        with pytest.raises(ValueError):
            betas_in,Ts_in = th.geometric_spaced_betas(n_chain,n_cold,T_cold,T_min,T_max,use_inf_final=True)

        return

    if T_cold == np.inf:
        with pytest.raises(AssertionError):
            betas_in,Ts_in = th.geometric_spaced_betas(n_chain,n_cold,T_cold,T_min,T_max,use_inf_final=True)
        return
    elif n_cold == n_chain:
        with pytest.warns(UserWarning):
            betas_in,Ts_in = th.geometric_spaced_betas(n_chain,n_cold,T_cold,T_min,T_max,use_inf_final=True)
            T_ladder = th.GeometricTemperatureLadder(n_chain,n_cold,T_cold,T_min,T_max,use_inf_final=True)
    else:
        betas_in,Ts_in = th.geometric_spaced_betas(n_chain,n_cold,T_cold,T_min,T_max,use_inf_final=True)
        T_ladder = th.GeometricTemperatureLadder(n_chain,n_cold,T_cold,T_min,T_max,use_inf_final=True)


    # Strict technical requirements
    assert Ts_in.size == n_chain                # check correct number of chains
    assert np.sum(Ts_in==T_cold) >= n_cold      # check correct number of cold chains
    assert not np.any(Ts_in < 0.)               # check no negative temperature chains
    assert np.all(T_ladder.Ts == Ts_in)         # check object matches
    assert np.all(T_ladder.betas[np.isfinite(Ts_in)&(Ts_in>0)]==1./Ts_in[np.isfinite(Ts_in)&(Ts_in>0)]) # check inverses match
    assert np.all(Ts_in[T_ladder.betas==0.] == np.inf) # check inverses match
    assert np.all(T_ladder.betas[Ts_in==0.] == np.inf) # check inverses match
    assert np.all(T_ladder.Ts == Ts_in)         # check object matches
    assert np.all(T_ladder.betas == betas_in)         # check object matches


    # Not technically required, but expected in this test case
    if not T_cold == np.inf:
        assert np.sum(Ts_in==T_cold) == n_cold                   # check correct number of cold chains
        if n_chain > n_cold:
            assert np.sum(Ts_in == np.inf) == 1                      # check 1 infinite temperature chain exists
        assert np.all(np.diff(Ts_in[n_cold:]) >= 0.)             # check non-cold chains are sorted
    else:
        assert np.sum(Ts_in==T_cold) == min(n_cold+1,n_chain)

    assert np.unique(Ts_in[n_cold:]).size == n_chain-n_cold  # check all temperatures are unique
    if T_cold <= 1.:
        assert np.all(Ts_in[n_cold:] > T_cold)                   # check all higher temperatures warmer than cold chain

@pytest.mark.parametrize("n_cold,n_chain,T_cold",test_set1)
def test_geometric_spacing(n_cold,n_chain,T_cold):
    """test the geoemtric based spacing produces results that makes sense"""
    T_min = 1.
    T_max = 1000.

    if n_cold > n_chain:
        with pytest.raises(ValueError):
            betas_in,Ts_in = th.geometric_spaced_betas(n_chain,n_cold,T_cold,T_min,T_max,use_inf_final=False)

        return

    if T_cold == np.inf:
        with pytest.raises(AssertionError):
            betas_in,Ts_in = th.geometric_spaced_betas(n_chain,n_cold,T_cold,T_min,T_max,use_inf_final=False)
        return
    else:
        betas_in,Ts_in = th.geometric_spaced_betas(n_chain,n_cold,T_cold,T_min,T_max,use_inf_final=False)
        T_ladder = th.GeometricTemperatureLadder(n_chain,n_cold,T_cold,T_min,T_max,use_inf_final=False)


    # Strict technical requirements
    assert Ts_in.size == n_chain                # check correct number of chains
    assert np.sum(Ts_in==T_cold) >= n_cold      # check correct number of cold chains
    assert not np.any(Ts_in < 0.)               # check no negative temperature chains
    assert np.all(T_ladder.betas[np.isfinite(Ts_in)&(Ts_in>0)]==1./Ts_in[np.isfinite(Ts_in)&(Ts_in>0)]) # check inverses match
    assert np.all(Ts_in[T_ladder.betas==0.] == np.inf) # check inverses match
    assert np.all(T_ladder.betas[Ts_in==0.] == np.inf) # check inverses match
    assert np.all(T_ladder.Ts == Ts_in)         # check object matches
    assert np.all(T_ladder.betas == betas_in)         # check object matches


    # Not technically required, but expected in this test case
    if not T_cold == np.inf:
        assert np.sum(Ts_in==T_cold) == n_cold                   # check correct number of cold chains
        if n_chain > n_cold:
            assert np.sum(Ts_in == np.inf) == 0                      # check no infinite temperature chain exists
        assert np.all(np.diff(Ts_in[n_cold:]) >= 0.)             # check non-cold chains are sorted
    else:
        assert np.sum(Ts_in==T_cold) == min(n_cold+1,n_chain)

    assert np.unique(Ts_in[n_cold:]).size == n_chain-n_cold  # check all temperatures are unique
    if T_cold <= 1.:
        assert np.all(Ts_in[n_cold:] > T_cold)                   # check all higher temperatures warmer than cold chain


if __name__=='__main__':
    pytest.cmdline.main(['tests/temperature_ladder_tests.py'])
