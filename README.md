# dspec
momentum spectrum decomposition

# Example of usage

## open a .nc file containing spectral data
```python
import datatree
file_L1b = '/home1/scratch/agrouaze/l1b/S1B_IW_SLC__1SDV_20210420T094142_20210420T094208_026549_032B99_7071.SAFE/s1b-iw3-slc-vv-20210420t094142-20210420t094208-026549-032b99-006_L1B_xspec_IFR_0.5.nc'
dt = datatree.open_datatree(file_L1b)
```


## symmetrize both imaginary and real part spectrum 
```python
import spectrum_momentum
one_spec_re = dt['intraburst_xspectra'].ds['xspectra_2tau_Re'].isel(burst=0,
                                                tile_sample=0,tile_line=0).mean(dim='2tau')
one_spec_im = dt['intraburst_xspectra'].ds['xspectra_2tau_Im'].isel(burst=0,
                                                tile_sample=0,tile_line=0).mean(dim='2tau')
print(one_spec_re)
one_spec_re = one_spec_re.swap_dims({'freq_sample':'k_rg','freq_line':'k_az'})
one_spec_re = spectrum_momentum.symmetrize_xspectrum(one_spec_re, dim_range='k_rg', dim_azimuth='k_az')
one_spec_im = one_spec_im.swap_dims({'freq_sample':'k_rg','freq_line':'k_az'})
one_spec_im = spectrum_momentum.symmetrize_xspectrum(one_spec_im, dim_range='k_rg', dim_azimuth='k_az')
one_spec_re
```

## compute 20 orthogonal momentum
```python
import spectrum_momentum
decomp_moments = spectrum_momentum.orthogonalDecompSpec(real_part_spectrum=one_spec_re,
                                                        imaginary_part_spectrum=one_spec_im)
decomp_moments
```