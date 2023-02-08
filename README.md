# dspec
momentum spectrum decomposition

# Example of usage

## open a .nc file containing spectral data
```python
import datatree
file_L1b = './S1B_IW_SLC__1SDV_20210420T094142_20210420T094208_026549_032B99_7071.SAFE/s1b-iw3-slc-vv-20210420t094142-20210420t094208-026549-032b99-006_L1B_xspec_IFR_0.5.nc'
dt = datatree.open_datatree(file_L1b)
```


## symmetrize both imaginary and real part spectrum 
```python
import spectrum_momentum
cat_xspec = 'intra'
ds = dt[cat_xspec+'burst_xspectra'].to_dataset()
for tautau in range(3):
    ds['xspectra_%stau'%tautau] = ds['xspectra_%stau_Re'%tautau] + 1j*ds['xspectra_%stau_Im'%tautau]
    ds = ds.drop(['xspectra_%stau_Re'%tautau,'xspectra_%stau_Im'%tautau])

xspec2tau  =  ds['xspectra_2tau'].swap_dims({'freq_sample':'k_rg','freq_line':'k_az'})
xspec2tau = spectrum_momentum.symmetrize_xspectrum(xspec2tau, dim_range='k_rg', dim_azimuth='k_az')
one_spec_re = xspec2tau.isel(burst=0,tile_sample=0,tile_line=0).mean(dim='2tau').real
one_spec_im = xspec2tau.isel(burst=0,tile_sample=0,tile_line=0).mean(dim='2tau').imag

print(one_spec_re)

```

## compute 20 orthogonal momentum (floats) associated to convolution of the spectra with weights
```python
import spectrum_momentum
decomp_moments = spectrum_momentum.orthogonalDecompSpec(real_part_spectrum=one_spec_re,
                                                        imaginary_part_spectrum=one_spec_im)
decomp_moments
```