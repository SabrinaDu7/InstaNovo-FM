# Paired outcomes: fork-greedy against casanovo-beam5

## By species

| group | spectra | both right | only fork-greedy | only casanovo-beam5 | neither | of casanovo-beam5's correct, fork-greedy also right | of fork-greedy's correct, casanovo-beam5 also right |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Apis-mellifera | 20,000 | 9,982 | 1,798 | 1,738 | 6,482 | 0.852 | 0.847 |
| Bacillus-subtilis | 20,000 | 11,777 | 1,572 | 1,938 | 4,713 | 0.859 | 0.882 |
| Candidatus-endoloripes | 20,000 | 8,796 | 1,589 | 1,445 | 8,170 | 0.859 | 0.847 |
| H.-sapiens | 20,000 | 11,689 | 1,346 | 1,628 | 5,337 | 0.878 | 0.897 |
| Methanosarcina-mazei | 20,000 | 11,668 | 1,750 | 1,748 | 4,834 | 0.870 | 0.870 |
| Mus-musculus | 20,000 | 9,375 | 1,622 | 1,638 | 7,365 | 0.851 | 0.853 |
| Saccharomyces-cerevisiae | 20,000 | 12,922 | 1,311 | 1,735 | 4,032 | 0.882 | 0.908 |
| Solanum-lycopersicum | 20,000 | 12,838 | 1,141 | 1,726 | 4,295 | 0.881 | 0.918 |
| Vigna-mungo | 20,000 | 12,289 | 2,174 | 1,846 | 3,691 | 0.869 | 0.850 |
| all | 180,000 | 101,336 | 14,303 | 15,442 | 48,919 | 0.868 | 0.876 |

## By mods

| group | spectra | both right | only fork-greedy | only casanovo-beam5 | neither | of casanovo-beam5's correct, fork-greedy also right | of fork-greedy's correct, casanovo-beam5 also right |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 123,925 | 82,243 | 10,679 | 7,237 | 23,766 | 0.919 | 0.885 |
| +0.984 | 14,855 | 1,782 | 328 | 3,275 | 9,470 | 0.352 | 0.845 |
| +15.995 | 9,096 | 4,692 | 986 | 570 | 2,848 | 0.892 | 0.826 |
| nterm+25.980 | 275 | 0 | 0 | 0 | 275 | nan | nan |
| nterm+42.011 | 717 | 154 | 21 | 140 | 402 | 0.524 | 0.880 |
| nterm+43.006 | 3,500 | 31 | 16 | 1,960 | 1,493 | 0.016 | 0.660 |
| nterm-17.027 | 3,995 | 1,145 | 403 | 711 | 1,736 | 0.617 | 0.740 |
| C+57.021 only | 18,773 | 11,038 | 1,752 | 1,165 | 4,818 | 0.905 | 0.863 |
| +0.984,+15.995 | 1,316 | 58 | 25 | 134 | 1,099 | 0.302 | 0.699 |
| +0.984,nterm+25.980 | 180 | 0 | 0 | 0 | 180 | nan | nan |
| +0.984,nterm+42.011 | 329 | 4 | 0 | 14 | 311 | 0.222 | 1.000 |
| +0.984,nterm+43.006 | 560 | 0 | 0 | 70 | 490 | 0.000 | nan |
| +0.984,nterm-17.027 | 563 | 20 | 8 | 66 | 469 | 0.233 | 0.714 |
| +15.995,nterm+42.011 | 814 | 4 | 56 | 14 | 740 | 0.222 | 0.067 |
| +15.995,nterm+43.006 | 229 | 0 | 2 | 28 | 199 | 0.000 | 0.000 |
| +15.995,nterm-17.027 | 454 | 163 | 26 | 58 | 207 | 0.738 | 0.862 |
| +0.984,+15.995,nterm+42.011 | 188 | 0 | 1 | 0 | 187 | nan | 0.000 |

## By n_peaks

| group | spectra | both right | only fork-greedy | only casanovo-beam5 | neither | of casanovo-beam5's correct, fork-greedy also right | of fork-greedy's correct, casanovo-beam5 also right |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <50 | 18,779 | 4,229 | 1,264 | 1,273 | 12,013 | 0.769 | 0.770 |
| 200+ | 90,293 | 53,566 | 8,423 | 8,150 | 20,154 | 0.868 | 0.864 |
| 50-99 | 31,043 | 17,141 | 2,132 | 2,430 | 9,340 | 0.876 | 0.889 |
| 100-199 | 39,885 | 26,400 | 2,484 | 3,589 | 7,412 | 0.880 | 0.914 |

## By length

| group | spectra | both right | only fork-greedy | only casanovo-beam5 | neither | of casanovo-beam5's correct, fork-greedy also right | of fork-greedy's correct, casanovo-beam5 also right |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30+ | 4,986 | 276 | 334 | 143 | 4,233 | 0.659 | 0.452 |
| <=9 | 18,621 | 15,441 | 640 | 1,620 | 920 | 0.905 | 0.960 |
| 10-14 | 72,921 | 48,952 | 4,860 | 7,001 | 12,108 | 0.875 | 0.910 |
| 15-19 | 51,743 | 27,307 | 4,899 | 4,395 | 15,142 | 0.861 | 0.848 |
| 20-29 | 31,729 | 9,360 | 3,570 | 2,283 | 16,516 | 0.804 | 0.724 |

