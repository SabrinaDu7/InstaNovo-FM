# Paired outcomes: fork-greedy against fork-scratch-greedy

## By species

| group | spectra | both right | only fork-greedy | only fork-scratch-greedy | neither | of fork-scratch-greedy's correct, fork-greedy also right | of fork-greedy's correct, fork-scratch-greedy also right |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Apis-mellifera | 5,000 | 2,880 | 222 | 318 | 1,580 | 0.901 | 0.928 |
| Bacillus-subtilis | 5,000 | 3,226 | 210 | 191 | 1,373 | 0.944 | 0.939 |
| Candidatus-endoloripes | 5,000 | 2,386 | 317 | 221 | 2,076 | 0.915 | 0.883 |
| H.-sapiens | 5,000 | 2,750 | 232 | 210 | 1,808 | 0.929 | 0.922 |
| Methanosarcina-mazei | 5,000 | 3,295 | 236 | 219 | 1,250 | 0.938 | 0.933 |
| Mus-musculus | 5,000 | 2,276 | 295 | 251 | 2,178 | 0.901 | 0.885 |
| Saccharomyces-cerevisiae | 5,000 | 3,468 | 161 | 199 | 1,172 | 0.946 | 0.956 |
| Solanum-lycopersicum | 5,000 | 3,507 | 238 | 226 | 1,029 | 0.939 | 0.936 |
| Vigna-mungo | 5,000 | 3,312 | 261 | 246 | 1,181 | 0.931 | 0.927 |
| all | 45,000 | 27,100 | 2,172 | 2,081 | 13,647 | 0.929 | 0.926 |

## By mods

| group | spectra | both right | only fork-greedy | only fork-scratch-greedy | neither | of fork-scratch-greedy's correct, fork-greedy also right | of fork-greedy's correct, fork-scratch-greedy also right |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 31,546 | 22,145 | 1,542 | 1,461 | 6,398 | 0.938 | 0.935 |
| +0.984 | 3,660 | 449 | 140 | 166 | 2,905 | 0.730 | 0.762 |
| +15.995 | 2,459 | 1,435 | 140 | 123 | 761 | 0.921 | 0.911 |
| nterm+42.011 | 159 | 45 | 8 | 3 | 103 | 0.938 | 0.849 |
| nterm+43.006 | 820 | 2 | 6 | 7 | 805 | 0.222 | 0.250 |
| nterm-17.027 | 913 | 267 | 81 | 82 | 483 | 0.765 | 0.767 |
| C+57.021 only | 4,316 | 2,678 | 236 | 215 | 1,187 | 0.926 | 0.919 |
| +0.984,+15.995 | 285 | 13 | 10 | 6 | 256 | 0.684 | 0.565 |
| +0.984,nterm+43.006 | 147 | 0 | 0 | 0 | 147 | nan | nan |
| +0.984,nterm-17.027 | 125 | 3 | 3 | 1 | 118 | 0.750 | 0.500 |
| +15.995,nterm+42.011 | 156 | 9 | 2 | 2 | 143 | 0.818 | 0.818 |
| +15.995,nterm-17.027 | 132 | 51 | 4 | 15 | 62 | 0.773 | 0.927 |

## By n_peaks

| group | spectra | both right | only fork-greedy | only fork-scratch-greedy | neither | of fork-scratch-greedy's correct, fork-greedy also right | of fork-greedy's correct, fork-scratch-greedy also right |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <50 | 4,952 | 1,235 | 283 | 217 | 3,217 | 0.851 | 0.814 |
| 200+ | 21,946 | 14,114 | 984 | 1,052 | 5,796 | 0.931 | 0.935 |
| 50-99 | 8,441 | 5,092 | 436 | 390 | 2,523 | 0.929 | 0.921 |
| 100-199 | 9,661 | 6,659 | 469 | 422 | 2,111 | 0.940 | 0.934 |

## By length

| group | spectra | both right | only fork-greedy | only fork-scratch-greedy | neither | of fork-scratch-greedy's correct, fork-greedy also right | of fork-greedy's correct, fork-scratch-greedy also right |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30+ | 1,025 | 100 | 29 | 24 | 872 | 0.806 | 0.775 |
| <=9 | 5,313 | 4,464 | 144 | 190 | 515 | 0.959 | 0.969 |
| 10-14 | 18,881 | 12,920 | 910 | 933 | 4,118 | 0.933 | 0.934 |
| 15-19 | 12,719 | 7,076 | 698 | 658 | 4,287 | 0.915 | 0.910 |
| 20-29 | 7,062 | 2,540 | 391 | 276 | 3,855 | 0.902 | 0.867 |

