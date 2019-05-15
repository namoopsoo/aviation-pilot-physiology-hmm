#### Quick disclaimer
I'm looking at  [this kaggle competition which Completed in 2019-February](https://www.kaggle.com/c/reducing-commercial-aviation-fatalities/data) , to see what I come up with. _(According to the Kaggle competitrion rules, I understand that publicly 
sharing code/data during a "live" competition can only be done if also sharing within the public forum. This particular 
competition has been over for a few months so once I'm done here, I will just aim to post a link to my repo in the 
discussion forums later on. But there is no sense of urgency as I see it.)_

#### The data, initial notes
The data  is `256Hz` sampled physiological data 
from `18` pilots, but split into `9` crews. If I understand the concept properly, the training and test data sets are from two
completely different scenarios and the intent is to try and predict another third even less related scenario! Wow.
The training data I understand was collected neither in a plane nor in a flight simulator, while pilots were subjected to distractions
of different kinds.

And the test set is from an actual flight simulator. 

**General thought** : The task here is to predict the probability of the different kinds of physiological states at every point in time.
But the real life will be in a plane and not a flight simulator, so who knows how effective this will be.

**Labeling**: I have no idea how the labels were produced for this data. It doesn't seem to be described. So for the different experiments, the data rows have the label `'event'`, which presumably come from pilot survey data? 

#### A General look
The test looks like nearly `18million` rows.

```bash
 $ wc *csv
 17965144 17965144 4786266819 test.csv
  4867422 4867422 1234764001 train.csv
 22832566 22832566 6021030820 total
 ```
 The test.csv also has an `experiment` column, which kind of confused me
 ```bash
(pandars3) $ head -2 train.csv
crew,experiment,time,seat,eeg_fp1,eeg_f7,eeg_f8,eeg_t4,eeg_t6,eeg_t5,eeg_t3,eeg_fp2,eeg_o1,eeg_p3,eeg_pz,eeg_f3,eeg_fz,eeg_f4,eeg_c4,eeg_p4,eeg_poz,eeg_c3,eeg_cz,eeg_o2,ecg,r,gsr,event
1,CA,0.01171875,1,-5.28545,26.775801,-9.52731,-12.7932,16.7178,33.737499,23.712299,-6.6958699999999975,29.2321,24.842899,3.92134,18.447001,1.07547,3.09029,37.368999,17.437599,19.2019,20.5968,-3.95115,14.5076,-4520.0,817.705994,388.829987,A

(pandars3) $ head -2 test.csv
id,crew,experiment,time,seat,eeg_fp1,eeg_f7,eeg_f8,eeg_t4,eeg_t6,eeg_t5,eeg_t3,eeg_fp2,eeg_o1,eeg_p3,eeg_pz,eeg_f3,eeg_fz,eeg_f4,eeg_c4,eeg_p4,eeg_poz,eeg_c3,eeg_cz,eeg_o2,ecg,r,gsr
0,1,LOFT,0.00000000,0,17.8995,6.12783,0.994807,-28.2062,-47.695499,-187.080002,-33.183498,-4.22078,8.17816,33.160301000000004,33.8125,21.744699,16.2938,-7.04448,-14.4051,-4.0338400000000005,-0.393799,31.8381,17.0756,-8.13735,-7323.120117,643.1770019999999,594.778992

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

* Looking at train.csv....
```python
df = pd.read_csv('train.csv')

 %time gpdf = df[['crew', 'experiment', 'seat', 'event']].groupby(by=['crew', 'experiment', 'seat', 'event']).size().reset_index()

gpdf = gpdf.rename(columns={0: 'count'}) 

gpdf.shape                                                             
# Out[40]: (108, 5)

In [44]: gpdf                                                                   
Out[44]: 
     crew experiment  seat event  count
0       1         CA     0     A   1713
1       1         CA     0     C  90418
2       1         CA     1     A   1733
3       1         CA     1     C  90435
4       1         DA     0     A  80003
5       1         DA     0     D  12074
6       1         DA     1     A  80985
7       1         DA     1     D  11145
8       1         SS     0     A  35726
9       1         SS     0     B   3837
10      1         SS     1     A  35731
11      1         SS     1     B   3852
12      2         CA     0     A    180
13      2         CA     0     C  91953
14      2         CA     1     A    164
15      2         CA     1     C  91935
16      2         DA     0     A  79097
17      2         DA     0     D  13097
18      2         DA     1     A  78157
19      2         DA     1     D  13942
20      2         SS     0     A  84450
21      2         SS     0     B   7681
22      2         SS     1     A  84522
23      2         SS     1     B   7690
24      3         CA     0     A    115
25      3         CA     0     C  92022
26      3         CA     1     A     55
27      3         CA     1     C  92040
28      3         DA     0     A  78694
29      3         DA     0     D  13439
..    ...        ...   ...   ...    ...
78      7         DA     1     A  78829
79      7         DA     1     D  13271
80      7         SS     0     A  84468
81      7         SS     0     B   7692
82      7         SS     1     A  84440
83      7         SS     1     B   7696
84      8         CA     0     A     92
85      8         CA     0     C  92016
86      8         CA     1     A     97
87      8         CA     1     C  92039
88      8         DA     0     A  78891
89      8         DA     0     D  13217
90      8         DA     1     A  78631
91      8         DA     1     D  13456
92      8         SS     0     A  83083
93      8         SS     0     B   7679
94      8         SS     1     A  83081
95      8         SS     1     B   7677
96     13         CA     0     A    166
97     13         CA     0     C  91996
98     13         CA     1     A    150
99     13         CA     1     C  91976
100    13         DA     0     A  77469
101    13         DA     0     D  14609
102    13         DA     1     A  78062
103    13         DA     1     D  14067
104    13         SS     0     A  84441
105    13         SS     0     B   7695
106    13         SS     1     A  84412
107    13         SS     1     B   7681

[108 rows x 5 columns]

# Oddly enough not every crew is represented hmm..
In [46]: print(gpdf.crew.unique().tolist())                                     
[1, 2, 3, 4, 5, 6, 7, 8, 13]

```
* So in the train data, every row is an experiment row, 
```python
In [49]: df.experiment.value_counts()                                           
Out[49]: 
DA    1658393
CA    1658376
SS    1550652
Name: experiment, dtype: int64

In [50]: df.shape[0] - df.experiment.value_counts().sum()                       
Out[50]: 0
```

#### What proportion of the experiments are crew members actually in the intended/expected states?
* So the groupby above can kind of help with this. 
* For instance, for the first person,
```python
In [55]: gpdf[(gpdf.crew == 1) & (gpdf.seat == 0)][['experiment', 'event', 'count']]                                                                   
Out[55]: 
  experiment event  count
0         CA     A   1713
1         CA     C  90418
4         DA     A  80003
5         DA     D  12074
8         SS     A  35726
9         SS     B   3837

In [73]: xdf['count'].sum()                                                     
Out[73]: 223771

def extract_proportions(xdf):
    return pd.Series({
    f'{event}/{experiment}': xdf[(xdf['experiment'] == experiment) & (xdf['event'] == event)].iloc[0]['count']/ 
                                  xdf[xdf['experiment'] == experiment]['count'].sum()
                                  
                                  for event in xdf.event.unique().tolist()
                                  for experiment in xdf.experiment.unique().tolist()
                                  if xdf[(xdf['experiment'] == experiment) & (xdf['event'] == event)].shape[0] != 0
    })

xdf =  gpdf[(gpdf.crew == 1) & (gpdf.seat == 0)][['experiment', 'event', 'count']]                                                      

In [70]: extract_proportions(xdf)                                               
Out[70]: 
A/CA    0.018593
A/DA    0.868871
A/SS    0.903015
C/CA    0.981407
D/DA    0.131129
B/SS    0.096985
dtype: float64

In [74]: 0.981407 + 0.018593, 0.131129+0.868871, 0.096985+0.903015              
Out[74]: (1.0, 1.0, 1.0)
```
* So just looking at this one particular person, we see for `223771` many rows,
* `DA (Diverted Attention)` is mostly  `A (Baseline)`, at `0.86` and otherwise `0.13` `D (Diverted)`,
* `CA (Channelized Attention)` is mostly `C ( =CA)`, at `0.98` , otherwise `A (Baseline)`
* `SS (Startle/Surprise)` is hmm mainly `A (Baseline)` , at `0.90` and otherwise just `0.09` in actual `B (SS)`.

##### Run this data for each person...
* Hmm, looking at at these `18` different (I think) pilots, wow there appears to be visibly not much distinction.
```python

statsdf = gpdf.groupby(by=['crew', 'seat']).apply(extract_proportions).reset_index()

In [78]: statsdf                                                                                                                                     
Out[78]: 
    crew  seat      A/CA      A/DA      A/SS      C/CA      D/DA      B/SS
0      1     0  0.018593  0.868871  0.903015  0.981407  0.131129  0.096985
1      1     1  0.018803  0.879030  0.902685  0.981197  0.120970  0.097315
2      2     0  0.001954  0.857941  0.916630  0.998046  0.142059  0.083370
3      2     1  0.001781  0.848619  0.916605  0.998219  0.151381  0.083395
4      3     0  0.001248  0.854135  0.916782  0.998752  0.145865  0.083218
5      3     1  0.000597  0.860974  0.916772  0.999403  0.139026  0.083228
6      4     0  0.001302  0.868853  0.916514  0.998698  0.131147  0.083486
7      4     1  0.001400  0.860968  0.916706  0.998600  0.139032  0.083294
8      5     0  0.001661  0.847193  0.916730  0.998339  0.152807  0.083270
9      5     1  0.001791  0.857766  0.916472  0.998209  0.142234  0.083528
10     6     0  0.002311  0.860514  0.916711  0.997689  0.139486  0.083289
11     6     1  0.001661  0.858872  0.916748  0.998339  0.141128  0.083252
12     7     0  0.001563  0.867075  0.916536  0.998437  0.132925  0.083464
13     7     1  0.001607  0.855907  0.916471  0.998393  0.144093  0.083529
14     8     0  0.000999  0.856505  0.915394  0.999001  0.143495  0.084606
15     8     1  0.001053  0.853877  0.915412  0.998947  0.146123  0.084588
16    13     0  0.001801  0.841341  0.916482  0.998199  0.158659  0.083518
17    13     1  0.001628  0.847312  0.916595  0.998372  0.152688  0.083405
```



#### Note on tooling thinking
* [keras vs pytorch](https://deepsense.ai/keras-or-pytorch/)

