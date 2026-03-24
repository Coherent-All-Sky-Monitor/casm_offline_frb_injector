# casm_offline_frb_injector

Synthetic FRB injection for CASM filterbank files.

## Install

```bash
source ~/software/dev/casm_venvs/casm_offline_env/bin/activate
cd ~/software/dev/casm_io && pip install --no-build-isolation -e .
cd ~/software/dev/casm_offline_frb_injector && pip install --no-build-isolation -e .
```

## inject-frb

Create a filterbank with a dispersed Gaussian pulse at a given DM, width, and S/N.

```bash
inject-frb -o my_frb.fil --dm 50 --fwhm 2 --snr 20
```

Multi-beam (64 beams, FRB in beam 0, beam-sequential layout for hella):

```bash
inject-frb -o multibeam.fil --dm 50 --fwhm 2 --snr 20 --nbeams 64 --ibeam 0
```

```python
from casm_offline_frb_injector import FRBInjector

injector = FRBInjector(dm=100, fwhm_samples=4, target_snr=30)
injector.write("output.fil")
```

Note: `inject-frb` calibrates SNR on float32 data before quantizing to uint8.
The actual SNR in the written file is ~65% of the target. Use
`inject-frb-dedigitized` for accurate uint8 SNR.

## inject-frb-dedigitized

Same pulse and SNR math as `inject-frb`, but adds the signal in continuous
space (via probabilistic de-digitization) and iteratively corrects the
amplitude until the target SNR is achieved on the actual uint8 output.

```bash
# Synthetic
inject-frb-dedigitized -o my_frb.fil --dm 100 --fwhm 5 --snr 20

# Into existing filterbank
inject-frb-dedigitized --input real_obs.fil -o injected.fil --dm 100 --fwhm 5 --snr 20

# Multi-beam
inject-frb-dedigitized -o multibeam.fil --dm 100 --fwhm 5 --snr 50 --nbeams 64 --ibeam 0
```

```python
from casm_offline_frb_injector.inject_frb_dedigitized import inject_synthetic, inject_into_file

data, metadata, header = inject_synthetic(dm=100, target_snr=20, fwhm_samples=5)
data, metadata, header = inject_into_file("real_obs.fil", dm=100, target_snr=20, fwhm_samples=5)
```

SNR recovery on uint8 data:

| Target S/N | `inject-frb` | `inject-frb-dedigitized` |
|-----------|-------------|------------------------|
| 10 | 6.5 | 9.8 |
| 20 | 13.1 | 19.7 |
| 50 | 32.9 | 49.7 |
| 100 | 68.4 | 99.9 |

Validation: `python tests/test_snr_recovery.py` or `pytest tests/test_snr_recovery.py -v`

## Plot

```python
from casm_io.filterbank import FilterbankFile
from casm_io.filterbank.plotting import plot_dedispersed_waterfall

fb = FilterbankFile("my_frb.fil")
plot_dedispersed_waterfall(fb.data, fb.header, dm=50, output_path="waterfall.png")
```

For multi-beam files:

```python
fb = FilterbankFile("multibeam.fil", beam=0)
plot_dedispersed_waterfall(fb.data, fb.header, dm=50, output_path="beam0.png")
```
