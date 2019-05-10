
#### The data, initial notes
The data from [kaggle](https://www.kaggle.com/c/reducing-commercial-aviation-fatalities/data) is `256Hz` sampled physiological data 
from `18` pilots, but split into `9` crews. If I understand the concept properly, the training and test data sets are from two
completely different scenarios and the intent is to try and predict another third even less related scenario! Wow.
The training data I understand was collected neither in a plane nor in a flight simulator, while pilots were subjected to distractions
of different kinds.

And the test set is from an actual flight simulator. The test looks like nearly `18million` rows.

```bash
 $ wc *csv
 17965144 17965144 4786266819 test.csv
  4867422 4867422 1234764001 train.csv
 22832566 22832566 6021030820 total
 ```
 The test.csv also has an `experiment` column, which kind of confused me
 ```bash
(pandars3) $ head -2 test.csv
id,crew,experiment,time,seat,eeg_fp1,eeg_f7,eeg_f8,eeg_t4,eeg_t6,eeg_t5,eeg_t3,eeg_fp2,eeg_o1,eeg_p3,eeg_pz,eeg_f3,eeg_fz,eeg_f4,eeg_c4,eeg_p4,eeg_poz,eeg_c3,eeg_cz,eeg_o2,ecg,r,gsr
0,1,LOFT,0.00000000,0,17.8995,6.12783,0.994807,-28.2062,-47.695499,-187.080002,-33.183498,-4.22078,8.17816,33.160301000000004,33.8125,21.744699,16.2938,-7.04448,-14.4051,-4.0338400000000005,-0.393799,31.8381,17.0756,-8.13735,-7323.120117,643.1770019999999,594.778992
(pandars3) $ head -2 train.csv
crew,experiment,time,seat,eeg_fp1,eeg_f7,eeg_f8,eeg_t4,eeg_t6,eeg_t5,eeg_t3,eeg_fp2,eeg_o1,eeg_p3,eeg_pz,eeg_f3,eeg_fz,eeg_f4,eeg_c4,eeg_p4,eeg_poz,eeg_c3,eeg_cz,eeg_o2,ecg,r,gsr,event
1,CA,0.01171875,1,-5.28545,26.775801,-9.52731,-12.7932,16.7178,33.737499,23.712299,-6.6958699999999975,29.2321,24.842899,3.92134,18.447001,1.07547,3.09029,37.368999,17.437599,19.2019,20.5968,-3.95115,14.5076,-4520.0,817.705994,388.829987,A
(pandars3) $ 
```
So I tried understanding if every row in `test.csv` has the same value `experiment=LOFT`, but my `ag` crashed heh, 
```
 $ ag LOFT -c 
ERR: expected to read 491299523 bytes but read 4294967295
```
But looking at the first 1M and last 1M, yep all have `LOFT`
```bash
(pandars3) $ head -1000000 test.csv  > test.first1M.csv
(pandars3) $ tail  -1000000 test.csv  > test.last1M.csv
(pandars3) $ ag LOFT -c test*1M.csv 
test.first1M.csv:999999
test.last1M.csv:1000000
```
Ok anyways, `grep` didnt crash though
```
(pandars3) $ time cat  test.csv| grep -c LOFT 
17965143

real	0m10.480s
user	0m9.116s
sys	0m3.685s
```
Ok what the heck, so what's the point of this column ? Every line has `experiment=LOFT` . 
Anyway at least indeed this means that's basically part of the unknown.


#### General thought.
The task here is to predict the probability of the different kinds of physiological states at every point in time.




