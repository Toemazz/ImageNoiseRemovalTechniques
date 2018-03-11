# ImageNoiseRemovalTechniques

## Description

### Part One
_PartOne.m_ evaluates the performance of the algorithms developed in https://github.com/Toemazz/ProductionLineVisualInspection in the presence of varying levels of noise. This involved choosing an appropriate range of (Gaussian) noise levels, plotting the fault detection performance for each function over the chosen range of noise levels and commenting on the relative performance of the system.

### Part Two
_PartTwo_A.m_ and _PartTwo_B.m_ carried out filtering on the images with noise before checking if the fault was detected and just like the first part, evaluate the fault detection performance of the algorithms. The objective being to reduce the number of errors caused by the noise by cleaning up the signals. Both spatial domain and frequency domain noise removal techniques were considered.
To compare the effectiveness of each noise removal technique a constant value of 0.5 was used for the Gaussian noise variance. This was noise variance value was chosen because it was over double the point where the ‘bottle underfilled’ algorithm achieved 0% fault detection rate. Each test was also run 10 times and the average system performance was plotted in each case.
