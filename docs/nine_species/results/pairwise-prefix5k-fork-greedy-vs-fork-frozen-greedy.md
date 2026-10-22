# Paired outcomes: fork-greedy against fork-frozen-greedy

## By species

| group | spectra | both right | only fork-greedy | only fork-frozen-greedy | neither | of fork-frozen-greedy's correct, fork-greedy also right | of fork-greedy's correct, fork-frozen-greedy also right |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Apis-mellifera | 5,000 | 2,502 | 600 | 224 | 1,674 | 0.918 | 0.807 |
| Bacillus-subtilis | 5,000 | 2,850 | 586 | 185 | 1,379 | 0.939 | 0.829 |
| Candidatus-endoloripes | 5,000 | 2,040 | 663 | 167 | 2,130 | 0.924 | 0.755 |
| H.-sapiens | 5,000 | 2,516 | 466 | 161 | 1,857 | 0.940 | 0.844 |
| Methanosarcina-mazei | 5,000 | 2,946 | 585 | 205 | 1,264 | 0.935 | 0.834 |
| Mus-musculus | 5,000 | 2,000 | 571 | 188 | 2,241 | 0.914 | 0.778 |
| Saccharomyces-cerevisiae | 5,000 | 3,132 | 497 | 155 | 1,216 | 0.953 | 0.863 |
| Solanum-lycopersicum | 5,000 | 3,372 | 373 | 274 | 981 | 0.925 | 0.900 |
| Vigna-mungo | 5,000 | 2,704 | 869 | 174 | 1,253 | 0.940 | 0.757 |
| all | 45,000 | 24,062 | 5,210 | 1,733 | 13,995 | 0.933 | 0.822 |

## By mods

| group | spectra | both right | only fork-greedy | only fork-frozen-greedy | neither | of fork-frozen-greedy's correct, fork-greedy also right | of fork-greedy's correct, fork-frozen-greedy also right |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 31,546 | 19,899 | 3,788 | 1,153 | 6,706 | 0.945 | 0.840 |
| +0.984 | 3,660 | 419 | 170 | 272 | 2,799 | 0.606 | 0.711 |
| +15.995 | 2,459 | 1,187 | 388 | 76 | 808 | 0.940 | 0.754 |
| nterm+42.011 | 159 | 40 | 13 | 2 | 104 | 0.952 | 0.755 |
| nterm+43.006 | 820 | 5 | 3 | 7 | 805 | 0.417 | 0.625 |
| nterm-17.027 | 913 | 150 | 198 | 55 | 510 | 0.732 | 0.431 |
| C+57.021 only | 4,316 | 2,310 | 604 | 140 | 1,262 | 0.943 | 0.793 |
| +0.984,+15.995 | 285 | 14 | 9 | 14 | 248 | 0.500 | 0.609 |
| +0.984,nterm+43.006 | 147 | 0 | 0 | 0 | 147 | nan | nan |
| +0.984,nterm-17.027 | 125 | 2 | 4 | 1 | 118 | 0.667 | 0.333 |
| +15.995,nterm+42.011 | 156 | 5 | 6 | 2 | 143 | 0.714 | 0.455 |
| +15.995,nterm-17.027 | 132 | 29 | 26 | 9 | 68 | 0.763 | 0.527 |

## By n_peaks

| group | spectra | both right | only fork-greedy | only fork-frozen-greedy | neither | of fork-frozen-greedy's correct, fork-greedy also right | of fork-greedy's correct, fork-frozen-greedy also right |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <50 | 4,952 | 1,067 | 451 | 173 | 3,261 | 0.860 | 0.703 |
| 200+ | 21,946 | 12,154 | 2,944 | 841 | 6,007 | 0.935 | 0.805 |
| 50-99 | 8,441 | 4,733 | 795 | 308 | 2,605 | 0.939 | 0.856 |
| 100-199 | 9,661 | 6,108 | 1,020 | 411 | 2,122 | 0.937 | 0.857 |

## By length

| group | spectra | both right | only fork-greedy | only fork-frozen-greedy | neither | of fork-frozen-greedy's correct, fork-greedy also right | of fork-greedy's correct, fork-frozen-greedy also right |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30+ | 1,025 | 50 | 79 | 7 | 889 | 0.877 | 0.388 |
| <=9 | 5,313 | 4,317 | 291 | 210 | 495 | 0.954 | 0.937 |
| 10-14 | 18,881 | 11,856 | 1,974 | 888 | 4,163 | 0.930 | 0.857 |
| 15-19 | 12,719 | 5,983 | 1,791 | 468 | 4,477 | 0.927 | 0.770 |
| 20-29 | 7,062 | 1,856 | 1,075 | 160 | 3,971 | 0.921 | 0.633 |

