# Data requirements and description
## Daily_Prec.txt
Multisite daily rainfall dataset to be disaggregated
* first 3 column: year, month and day
* other columnd: daily rainfall series in site order

## CLA_ERA20C_*_mslp-anom_1950-2010.cla
Circulation pattern (CP) classifications, where * represents the number of CP classes 
* file type: Tab-separated values file without header
* first 3 column: year, month and day
* 4th column: can be ignored
* 5th column: CP classes; datatype: integer

## Hourly_Prec_his.txt
Historical observed multisite hourly rainfall dataset
* the number of rainfall sites should be the same as that in Daily_Prec.txt
* first 4 column: year, month, day and hour
* the other columns: hourly rainfall data series
## Rainfall_site_lookup.txt
Information on rainfall observation sites, including ID, station names, spatial cooridinates, site orders and indexes. 
File extension should be .txt or .csv (normal text file)
