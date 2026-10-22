# Paired outcomes: fork-beam5 against fork-greedy

## By species

| group | spectra | both right | only fork-beam5 | only fork-greedy | neither | of fork-greedy's correct, fork-beam5 also right | of fork-beam5's correct, fork-greedy also right |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Apis-mellifera | 5,000 | 3,079 | 310 | 23 | 1,588 | 0.993 | 0.909 |
| Bacillus-subtilis | 5,000 | 3,418 | 211 | 18 | 1,353 | 0.995 | 0.942 |
| Candidatus-endoloripes | 5,000 | 2,671 | 138 | 32 | 2,159 | 0.988 | 0.951 |
| H.-sapiens | 5,000 | 2,948 | 137 | 34 | 1,881 | 0.989 | 0.956 |
| Methanosarcina-mazei | 5,000 | 3,502 | 172 | 29 | 1,297 | 0.992 | 0.953 |
| Mus-musculus | 5,000 | 2,497 | 152 | 74 | 2,277 | 0.971 | 0.943 |
| Saccharomyces-cerevisiae | 5,000 | 3,599 | 164 | 30 | 1,207 | 0.992 | 0.956 |
| Solanum-lycopersicum | 5,000 | 3,721 | 220 | 24 | 1,035 | 0.994 | 0.944 |
| Vigna-mungo | 5,000 | 3,540 | 238 | 33 | 1,189 | 0.991 | 0.937 |
| all | 45,000 | 28,975 | 1,742 | 297 | 13,986 | 0.990 | 0.943 |

## By mods

| group | spectra | both right | only fork-beam5 | only fork-greedy | neither | of fork-greedy's correct, fork-beam5 also right | of fork-beam5's correct, fork-greedy also right |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 31,546 | 23,575 | 1,239 | 112 | 6,620 | 0.995 | 0.950 |
| +0.984 | 3,660 | 578 | 138 | 11 | 2,933 | 0.981 | 0.807 |
| +15.995 | 2,459 | 1,567 | 125 | 8 | 759 | 0.995 | 0.926 |
| nterm+42.011 | 159 | 53 | 2 | 0 | 104 | 1.000 | 0.964 |
| nterm+43.006 | 820 | 1 | 1 | 7 | 811 | 0.125 | 0.500 |
| nterm-17.027 | 913 | 219 | 43 | 129 | 522 | 0.629 | 0.836 |
| C+57.021 only | 4,316 | 2,899 | 167 | 15 | 1,235 | 0.995 | 0.946 |
| +0.984,+15.995 | 285 | 21 | 9 | 2 | 253 | 0.913 | 0.700 |
| +0.984,nterm+43.006 | 147 | 0 | 0 | 0 | 147 | nan | nan |
| +0.984,nterm-17.027 | 125 | 3 | 1 | 3 | 118 | 0.500 | 0.750 |
| +15.995,nterm+42.011 | 156 | 9 | 5 | 2 | 140 | 0.818 | 0.643 |
| +15.995,nterm-17.027 | 132 | 48 | 11 | 7 | 66 | 0.873 | 0.814 |

## By n_peaks

| group | spectra | both right | only fork-beam5 | only fork-greedy | neither | of fork-greedy's correct, fork-beam5 also right | of fork-beam5's correct, fork-greedy also right |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <50 | 4,952 | 1,470 | 144 | 48 | 3,290 | 0.968 | 0.911 |
| 200+ | 21,946 | 14,990 | 974 | 108 | 5,874 | 0.993 | 0.939 |
| 50-99 | 8,441 | 5,450 | 227 | 78 | 2,686 | 0.986 | 0.960 |
| 100-199 | 9,661 | 7,065 | 397 | 63 | 2,136 | 0.991 | 0.947 |

## By length

| group | spectra | both right | only fork-beam5 | only fork-greedy | neither | of fork-greedy's correct, fork-beam5 also right | of fork-beam5's correct, fork-greedy also right |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30+ | 1,025 | 129 | 22 | 0 | 874 | 1.000 | 0.854 |
| <=9 | 5,313 | 4,572 | 195 | 36 | 510 | 0.992 | 0.959 |
| 10-14 | 18,881 | 13,685 | 807 | 145 | 4,244 | 0.990 | 0.944 |
| 15-19 | 12,719 | 7,692 | 490 | 82 | 4,455 | 0.989 | 0.940 |
| 20-29 | 7,062 | 2,897 | 228 | 34 | 3,903 | 0.988 | 0.927 |

