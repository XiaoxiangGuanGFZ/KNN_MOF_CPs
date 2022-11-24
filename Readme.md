# kNN MOF CP
kNN: k-nearest neighbors sampling 

MOF: method of fragments

CP: circulation patterns
## Introduction
A temporal disaggregation model is usually used to derive sub-daily (like hourly) distributions of weather variables from daily scale. kNN_MOF_CP is a non-parametric dtsaggregation procedure developed for multisite daily-to-hourly rainfall and air temperature disaggregation. In the procedure, mothod of fragments (MOF) based on k-nearest neighbor resampling conditioned on circulation pattern classification is applied. 
## Techniques
### Method Of Fragements
The idea of MOF is to resample a vector of fragments that represents the relative distribution of subdaily to daily rainfall (Pui et al., 2012). The number of fragments corresponds to the subdaily temporal resolution used, i.e. if the disaggregation is conducted from daily to hourly resolution, the relative distribution of subdaily values consists of 24 relative weights that sum up to 1
### k-Nearest Neighbor algorithm
The k-nearest neighbors algorithm, also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point.
### Circulation pattern classification
circulation patterns are divided on the basis of daily values of the mean sea-level pressure field of the ERA5 data set for the period from January 1979 to July 2021. 
The classification is conducted using the objective classification algorithm SANDRA (Simulated ANnealing and Diversified RAndomization) (Philipp et al., 2007). It is based on k-means and minimizes the within-cluster variance of the Euclidian distance between the cluster elements and the cluster centroid. 
## How to use

## Paper related
Xiaoxiang Guan, Katrin Nissen, Dung Viet Nguyen, Bruno Merz, Benjamin Winter, Sergiy Vorogushyn. Multisite temporal rainfall disaggregation using methods of fragments conditioned on circulation patterns. (under review)

## References
Pui, A., Sharma, A., Mehrotra, R., Sivakumar, B. and Jeremiah, E.  2012.  A comparison of alternatives for daily to sub-daily rainfall disaggregation. Journal of Hydrology, 470-471, 138-157. doi: https://doi.org/10.1016/j.jhydrol.2012.08.041.

Philipp, A., Della-Marta, P.M., Jacobeit, J., Fereday, D.R., Jones, P.D., Moberg, A. and Wanner, H.  2007.  Long-Term Variability of Daily North Atlantic–European Pressure Patterns since 1850 Classified by Simulated Annealing Clustering. Journal of Climate, 20(16), 4065-4095. doi: https://doi.org/10.1175/jcli4175.1.

## Author
[Xiaoxiang Guan](https://www.gfz-potsdam.de/staff/guan.xiaoxiang/sec44)

Email: guan@gfz-potsdam.de


