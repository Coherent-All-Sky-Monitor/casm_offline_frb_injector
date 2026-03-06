# casm_offline_frb_injector

Offline FRB injection toolkit for CASM
Generates filterbank files containing dispersed Gaussian pulses in calibrated
white noise, then optionally searches them with Hella to measure recovery.

## What This Does

1. **`inject_frb.py`** -- Create a single `.fil` file with a fake FRB at a
   chosen DM, width, and S/N.
2. **`batch_inject_frbs.py`** -- Generate thousands of injections with random
   parameters drawn from configurable distributions.
3. **`run_hella.py`** -- Run the Hella search pipeline on the injections and
   compare recovered candidates against ground truth.

## Prerequisites

```bash
source ~/software/dev/casm_venvs/casm_offline_env/bin/activate
cd ~/software/dev/casm_io && pip install -e .
```

This gives you `casm_io`, which provides the filterbank writer, reader, and
plotting tools used throughout.

## Quick Start

### 1. Inject a single FRB

```bash
python inject_frb.py \
    -o my_frb.fil \
    --dm 50 \
    --fwhm 2.0 \
    --snr 20 \
    --seed 42
```

This creates `my_frb.fil` -- a filterbank file containing a Gaussian pulse
dispersed to DM = 50 pc/cm^3, with intrinsic FWHM of 2 samples (~2.1 ms) and
a matched-filter S/N of ~20. The output tells you what happened:

```
Creating Gaussian burst filterbank
  DM: 50.0 pc/cm^3
  Intrinsic FWHM: 2.0 samples (2.097 ms)
  Target S/N: 20.0
  Dispersion sweep: 506 samples (530.6 ms)
  Frequency range: 375.031 - 468.750 MHz
  Channels: 3072, Samples: 4096
  ...
  Measured matched-filter S/N: 21.2
  Written to my_frb.fil
```

### 2. Plot the dedispersed waterfall

```python
from casm_io.filterbank import FilterbankFile
from casm_io.filterbank.plotting import plot_dedispersed_waterfall

fb = FilterbankFile("my_frb.fil")
plot_dedispersed_waterfall(fb.data, fb.header, dm=50, output_path="my_frb_waterfall.png")
```

The top panel shows the dedispersed timeseries (you should see a spike at the
pulse location) and the bottom panel shows the frequency-time waterfall.

### 3. Use the classes from Python

```python
from inject_frb import FRBInjector

injector = FRBInjector(dm=100, fwhm_samples=4.0, target_snr=30, seed=123)
result = injector.inject()

print(result["measured_snr"])   # actual S/N achieved
print(result["pulse_center"])   # sample index of pulse at highest freq
print(result["data"].shape)     # (4096, 3072) uint8 array

injector.write("output.fil")    # write to disk
```

## Batch Injection

Generate 2000 FRBs with random DM, S/N, and width:

```bash
python batch_inject_frbs.py \
    --outdir /data/casm/injections_2k \
    --ninj 2000
```

This writes one `.fil` per injection plus `injections_manifest.csv` with all
parameters and metadata. Dry-run first to check your commands:

```bash
python batch_inject_frbs.py \
    --outdir /data/casm/injections_2k \
    --ninj 5 \
    --dry_run
```

### Default parameter distributions

| Parameter | Range | Distribution |
|-----------|-------|-------------|
| DM | 20 -- 2000 pc/cm^3 | uniform |
| S/N | 6.5 -- 65 | uniform |
| Width | 0.262 -- 20 ms | log-uniform |
| Position | 0.1 -- 0.9 | uniform |

All ranges and distributions are CLI-overridable. See `--help` for all options.

## Running Hella

After generating injections, search them with Hella and evaluate recovery:

```bash
python run_hella.py \
    --input_dir /data/casm/injections_2k \
    --gpus 0,1 \
    --versions_to_run v1
```

This creates per-version summary CSVs in `hella_summaries/` with detection
rates, S/N recovery fractions, and DM accuracy.

The runner is resumable -- if interrupted, re-run the same command and it
picks up where it left off.

## CASM Instrument Defaults

These are the default filterbank parameters matching the CASM correlator
output. All are overridable via `--fch1`, `--foff`, etc.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fch1` | 468.75 MHz | Top of band |
| `foff` | -0.030517578125 MHz | Channel width (125/4096, descending) |
| `nchans` | 3072 | Number of frequency channels |
| `tsamp` | 0.001048576 s | Sampling time (32.768 us x 32) |
| `nsamples` | 4096 | Time samples per file (~4.29 s) |
| `nbits` | 8 | Bits per sample |




