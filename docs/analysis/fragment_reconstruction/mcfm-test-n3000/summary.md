# Fragment-ion reconstruction outside the framework (/home/sabrina/Documents/experiments/datasets/instanovo_fm/mcfm/mcfm-test-00000-of-00014.parquet, 3000 spectra, 3 blur seeds)

Produced by `scripts/analysis/fragment_reconstruction.py` (commit c04a5d3). Bin correct = predicted group and offset both exact (0.2 Da bin); ppm = |bin centre - true m/z| / true m/z. Greedy = the IG task's decoding; top-3 = the trainer's joint decoding for its ppm metrics; teacher = the offset head conditioned on the true group, as in the trainer's bin accuracy. Group = the whole fragment group masked (the IG task); base = the base ion alone masked.

Masked groups per model and regime: 42,441 (2,715 spectra with at least one b/y group).

## A. Bin accuracy by masking regime and decoder (mean over blur seeds; ± = spread of the per-seed accuracies)

| model | group / greedy | group / top3 | group / teacher | base / greedy | base / top3 | base / teacher |
|---|---|---|---|---|---|---|
| 40M | 42.1 % ± 0.1 | 42.6 % ± 0.1 | 42.1 % ± 0.1 | 43.6 % ± 0.1 | 44.2 % ± 0.1 | 43.6 % ± 0.1 |
| 40M-ours | 37.5 % ± 0.1 | 38.1 % ± 0.1 | 37.5 % ± 0.1 | 39.1 % ± 0.1 | 39.7 % ± 0.0 | 39.1 % ± 0.1 |
| 89M | 52.1 % ± 0.1 | 52.6 % ± 0.1 | 52.1 % ± 0.1 | 51.6 % ± 0.2 | 52.2 % ± 0.1 | 51.6 % ± 0.2 |

## B. Difference against 40M (points of bin accuracy, 95 % interval from resampling spectra; seeds pooled)

| model | group / greedy | group / top3 | group / teacher | base / greedy | base / top3 | base / teacher |
|---|---|---|---|---|---|---|
| 40M-ours | -4.4 [-4.8, -4.0] | -4.4 [-4.8, -4.0] | -4.4 [-4.8, -4.0] | -4.3 [-4.8, -3.9] | -4.2 [-4.6, -3.8] | -4.3 [-4.8, -3.9] |
| 89M | +9.7 [+9.1, +10.3] | +9.7 [+9.1, +10.3] | +9.7 [+9.1, +10.3] | +7.8 [+7.3, +8.4] | +7.9 [+7.4, +8.4] | +7.8 [+7.3, +8.4] |

## C. Group regime, greedy (the IG task's setting): by ion type and group size; median ppm from the bin centre

| model | b | y | single-peak groups | multi-peak groups | group accuracy | median ppm (all) | median ppm (correct bin) | median ppm (wrong bin) |
|---|---|---|---|---|---|---|---|---|
| 40M | 36.6 % (n=15,185) | 45.2 % (n=27,256) | 32.4 % (n=12,871) | 46.3 % (n=29,570) | 48.0 % | 13057 | 73 | 200130 |
| 40M-ours | 30.4 % (n=15,185) | 41.5 % (n=27,256) | 26.4 % (n=12,871) | 42.4 % (n=29,570) | 44.7 % | 18102 | 73 | 143354 |
| 89M | 48.2 % (n=15,185) | 54.3 % (n=27,256) | 47.6 % (n=12,871) | 54.0 % (n=29,570) | 55.4 % | 245 | 77 | 349035 |

## D. The trainer's regime (span masking with isotope co-masking, every masked peak)

| model | decoder | annotated: bin / group | unannotated: bin / group | all: bin / group | share annotated | median ppm all / annotated |
|---|---|---|---|---|---|---|
| 40M | greedy | 58.0 / 78.2 % | 23.2 / 59.6 % | 32.5 / 64.6 % | 26.7 % | 3084 / 127 |
| 40M | top3 | 58.7 / 78.1 % | 23.6 / 57.9 % | 33.0 / 63.3 % | 26.7 % | 3009 / 124 |
| 40M | teacher | 58.0 / 78.2 % | 23.2 / 59.6 % | 32.5 / 64.6 % | 26.7 % | 3605 / 127 |
| 40M-ours | greedy | 53.3 / 75.6 % | 21.6 / 58.8 % | 30.1 / 63.2 % | 26.7 % | 3650 / 168 |
| 40M-ours | top3 | 54.0 / 75.4 % | 22.1 / 57.1 % | 30.6 / 62.0 % | 26.7 % | 3554 / 159 |
| 40M-ours | teacher | 53.3 / 75.6 % | 21.6 / 58.8 % | 30.1 / 63.2 % | 26.7 % | 4314 / 169 |
| 89M | greedy | 73.8 / 86.7 % | 32.9 / 65.7 % | 43.8 / 71.3 % | 26.7 % | 734 / 90 |
| 89M | top3 | 74.5 / 86.8 % | 33.4 / 63.9 % | 44.4 / 70.0 % | 26.7 % | 669 / 89 |
| 89M | teacher | 73.8 / 86.7 % | 32.9 / 65.7 % | 43.8 / 71.3 % | 26.7 % | 755 / 90 |
