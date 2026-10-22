# Paired outcomes: fork-greedy against casanovo-greedy

## By species

| group | spectra | both right | only fork-greedy | only casanovo-greedy | neither | of casanovo-greedy's correct, fork-greedy also right | of fork-greedy's correct, casanovo-greedy also right |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Apis-mellifera | 20,000 | 9,560 | 2,220 | 1,390 | 6,830 | 0.873 | 0.812 |
| Bacillus-subtilis | 20,000 | 11,385 | 1,964 | 1,658 | 4,993 | 0.873 | 0.853 |
| Candidatus-endoloripes | 20,000 | 8,506 | 1,879 | 1,222 | 8,393 | 0.874 | 0.819 |
| H.-sapiens | 20,000 | 11,421 | 1,614 | 1,354 | 5,611 | 0.894 | 0.876 |
| Methanosarcina-mazei | 20,000 | 11,241 | 2,177 | 1,544 | 5,038 | 0.879 | 0.838 |
| Mus-musculus | 20,000 | 9,074 | 1,923 | 1,437 | 7,566 | 0.863 | 0.825 |
| Saccharomyces-cerevisiae | 20,000 | 12,638 | 1,595 | 1,375 | 4,392 | 0.902 | 0.888 |
| Solanum-lycopersicum | 20,000 | 12,642 | 1,337 | 1,505 | 4,516 | 0.894 | 0.904 |
| Vigna-mungo | 20,000 | 11,796 | 2,667 | 1,621 | 3,916 | 0.879 | 0.816 |
| all | 180,000 | 98,263 | 17,376 | 13,106 | 51,255 | 0.882 | 0.850 |

## By mods

| group | spectra | both right | only fork-greedy | only casanovo-greedy | neither | of casanovo-greedy's correct, fork-greedy also right | of fork-greedy's correct, casanovo-greedy also right |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 123,925 | 79,890 | 13,032 | 6,225 | 24,778 | 0.928 | 0.860 |
| +0.984 | 14,855 | 1,718 | 392 | 2,514 | 10,231 | 0.406 | 0.814 |
| +15.995 | 9,096 | 4,470 | 1,208 | 457 | 2,961 | 0.907 | 0.787 |
| nterm+25.980 | 275 | 0 | 0 | 0 | 275 | nan | nan |
| nterm+42.011 | 717 | 146 | 29 | 123 | 419 | 0.543 | 0.834 |
| nterm+43.006 | 3,500 | 26 | 21 | 1,850 | 1,603 | 0.014 | 0.553 |
| nterm-17.027 | 3,995 | 1,112 | 436 | 657 | 1,790 | 0.629 | 0.718 |
| C+57.021 only | 18,773 | 10,662 | 2,128 | 975 | 5,008 | 0.916 | 0.834 |
| +0.984,+15.995 | 1,316 | 55 | 28 | 101 | 1,132 | 0.353 | 0.663 |
| +0.984,nterm+25.980 | 180 | 0 | 0 | 0 | 180 | nan | nan |
| +0.984,nterm+42.011 | 329 | 4 | 0 | 13 | 312 | 0.235 | 1.000 |
| +0.984,nterm+43.006 | 560 | 0 | 0 | 63 | 497 | 0.000 | nan |
| +0.984,nterm-17.027 | 563 | 20 | 8 | 56 | 479 | 0.263 | 0.714 |
| +15.995,nterm+42.011 | 814 | 3 | 57 | 11 | 743 | 0.214 | 0.050 |
| +15.995,nterm+43.006 | 229 | 0 | 2 | 19 | 208 | 0.000 | 0.000 |
| +15.995,nterm-17.027 | 454 | 155 | 34 | 42 | 223 | 0.787 | 0.820 |
| +0.984,+15.995,nterm+42.011 | 188 | 0 | 1 | 0 | 187 | nan | 0.000 |

## By n_peaks

| group | spectra | both right | only fork-greedy | only casanovo-greedy | neither | of casanovo-greedy's correct, fork-greedy also right | of fork-greedy's correct, casanovo-greedy also right |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <50 | 18,779 | 4,087 | 1,406 | 1,078 | 12,208 | 0.791 | 0.744 |
| 200+ | 90,293 | 51,647 | 10,342 | 6,773 | 21,531 | 0.884 | 0.833 |
| 50-99 | 31,043 | 16,749 | 2,524 | 2,138 | 9,632 | 0.887 | 0.869 |
| 100-199 | 39,885 | 25,780 | 3,104 | 3,117 | 7,884 | 0.892 | 0.893 |

## By length

| group | spectra | both right | only fork-greedy | only casanovo-greedy | neither | of casanovo-greedy's correct, fork-greedy also right | of fork-greedy's correct, casanovo-greedy also right |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30+ | 4,986 | 231 | 379 | 88 | 4,288 | 0.724 | 0.379 |
| <=9 | 18,621 | 15,215 | 866 | 1,521 | 1,019 | 0.909 | 0.946 |
| 10-14 | 72,921 | 47,609 | 6,203 | 6,212 | 12,897 | 0.885 | 0.885 |
| 15-19 | 51,743 | 26,379 | 5,827 | 3,666 | 15,871 | 0.878 | 0.819 |
| 20-29 | 31,729 | 8,829 | 4,101 | 1,619 | 17,180 | 0.845 | 0.683 |

