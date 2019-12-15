# Data

This directory contains the datasets analysed both within the paper and the supplementary materials.

## Data format

In order for the MCMC code to perform correctly the data must be set up in the correct format. The data files are required to be in space separated value format (SSV) and contain data over n rows and K columns, where n is the number of observations (rankers) and K is the total number of entities within the data. The observations must be in preference order, that is, column 1 must contain the most preferred entity, column 2 the second most preferred entity and so on. The entities should be labelled numerically from 1 to K. For example, an entry of

1 2 3 4 5 6

within row i of the data file would specify that ranker i preferred entity 1 over entity 2, and entity 2 over entity 3 and so on.

**Note:** The Extended Plackett-Luce model is only defined for complete rankings which occur when a ranker considers and ranks all possible entities.






