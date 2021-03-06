
#### Quick look at the `time` field here, 
* So the `train.csv` data describes several experiments on several crews I believe.
* At least initially, the time doesn't look sequential,
*  It ends at `99seconds` , but the max `time` is `360seconds`
```python
df = pd.read_csv('data/train.csv')
print(df.time.iloc[-5:].tolist())
#    [99.991005, 99.993004, 99.994003, 99.997002, 99.998001]

df.time.tolist() == sorted(df.time.tolist())
#    False

df.time.iloc[:10].tolist() == sorted(df.time.iloc[:10].tolist())
#    True

df.time.max(), df.time.min()
#    (360.37109375, 0.003)
```
* What the heck.. I dont understand why even for a particular individual and experiment, the time is not sorted.
```python
(df[(df.crew == 1)&(df.seat == 1)& (df.experiment == 'CA')].time.tolist()
 == sorted(df[(df.crew == 1)&(df.seat == 1)& (df.experiment == 'CA')].time.tolist())
)
# False

# Script to see when is first occurrence of this
lasttime = 0
for i,time in enumerate(df.time.tolist()):
    if time < lasttime:
        print(i, lasttime, time)
        break
    else:
        lasttime = time
        
# => i, lasttime, time
# 6606 109.9960938 11.0

pd.concat([
        df[['crew', 'seat', 'time', 'r', 'experiment', 'event']].iloc[:3],
        df[['crew', 'seat', 'time', 'r', 'experiment', 'event']].iloc[6600:6610]

])
# 
```
* so weird, so at `i=6606`, why is the time jumping around? 
```
	crew	seat	time	r	experiment	event
0	1	1	0.011719	817.705994	CA	A
1	1	1	0.015625	817.705994	CA	A
2	1	1	0.019531	817.705994	CA	A
6600	1	1	109.988281	817.437988	CA	C
6601	1	0	109.988281	664.265991	CA	C
6602	1	0	109.992188	664.265991	CA	C
6603	1	1	109.992188	817.442017	CA	C
6604	1	1	109.996094	817.442017	CA	C
6605	1	0	109.996094	664.265991	CA	C
6606	1	0	11.000000	664.331970	CA	C
6607	1	1	11.000000	817.898987	CA	C
6608	1	0	11.003906	664.331970	CA	C
6609	1	1	11.003906	817.898987	CA	C
```
* , And each group with like 92k observations each.
```python
choices = ([1,2], [0,1], ['CA', 'DA', 'SS'])
for crew, seat, experiment in itertools.product(*choices):
    query = (df.crew == crew)&(df.seat == seat)& (df.experiment == experiment)
    print(df[query].shape)
    
(92131, 28)
(92077, 28)
(39563, 28)
(92168, 28)
(92130, 28)
(39583, 28)
(92133, 28)
(92194, 28)
(92131, 28)
(92099, 28)
(92099, 28)
(92212, 28)
```

#### Ok something periodic happening in this data I dont know yet..
```python
def what_records(dfx):
    return [(int(x), y) for (x,y) in json.loads(dfx['time'].to_json(orient='columns')).items()]
    
def calc_bumps(df):
    lasttime = 0
    bumps = []
    bumps_per_group = []
    choices = ([1], [0], ['CA', 'DA'])
    choices = ([1,2,3,4,5,6,7,8], [0,1], ['CA', 'DA', 'SS'])
    for crew, seat, experiment in itertools.product(*choices):
        query = (df.crew == crew)&(df.seat == seat)& (df.experiment == experiment)
        # for i, time in enumerate(df[query].time.tolist()):
        for i, time in what_records(df[query]):
            if time < lasttime:
                bumps.append((crew, seat, experiment, i, lasttime, time))

            lasttime = time

        bumps_per_group.append({'crew': crew, 'seat': seat, 'experiment': experiment, 
                                'bumps': len(bumps)})    
        lasttime = 0
        bumps = []

    return pd.DataFrame.from_records(bumps_per_group)

%time bumpsdf = calc_bumps(df)
# Wall time: 24.9 s
```
* This unsortedness has a regularity. Each group gets it like `32` or `33` times. umm..
* Is it possible like the experiment is repeated `32` or `33` times for each crew? This is not described in the notes. 
```python
bumpsdf.bumps.value_counts()
32    29
33    17
14     2
Name: bumps, dtype: int64
```

#### Contiguities
* umm... so weird.. the time jumps around so much. I didnt expect the data to be so dirty. Or at least a slight explanation would have helped.
```
contiguous = []
lasttime = 0; start = 0; end = None
for i, time in what_records(df.iloc[:1000000]):
    if time < lasttime:
        # stop
        end = lasttime
        contiguous.append([i, start, end])
        start = time
    lasttime = time
        
[[6606, 0, 109.9960938],
 [12238, 11.0, 119.9960938],
 [17870, 12.0, 129.9960938],
 [23502, 13.0, 139.9960938],
 [29134, 14.0, 149.9960938],
 [34766, 15.0, 159.9960938],
 [40398, 16.0, 169.9960938],
 [46030, 17.0, 179.9960938],
 [51662, 18.0, 189.9960938],
 [57294, 19.0, 199.9960938],
 [63438, 2.0, 209.9960938],
 [69070, 21.0, 219.9960938],
 [74702, 22.0, 229.9960938],
 [80334, 23.0, 239.9960938],
 [85966, 24.0, 249.9960938],
 [91598, 25.0, 259.9960938],
 [97230, 26.0, 269.9960938],
 [102862, 27.0, 279.9960938],
 [108494, 28.0, 289.9960938],
 [114126, 29.0, 299.9960938]]
```

#### I guess strictly speaking are there any duplications in the time variable?
* I Should have checked earlier, but looking at the Discussion, one person left a [comment](https://www.kaggle.com/c/reducing-commercial-aviation-fatalities/discussion/76681#latest-458859) about this too.. 
* They plotted time time variable for `crew=1, seat=0` , finding this, 
<img src="https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/assets/Screen%20Shot%202019-05-14%20at%205.17.02%20PM.png" width="530" height="350">

* [source: discussion thread](https://www.kaggle.com/c/reducing-commercial-aviation-fatalities/discussion/76681#latest-458859) .

* Okay so this is for one person, and it looks like indeed the data is just unsorted
* so if I can quickly look for any duplicates for all the people, then a sort should do the trick!

```python
def calc_duplicates(df):
    duplicates = []
    choices = ([1,2,3,4,5,6,7,8], [0,1], ['CA', 'DA', 'SS'])
    for crew, seat, experiment in itertools.product(*choices):
        query = (df.crew == crew)&(df.seat == seat)& (df.experiment == experiment)

        raw_count, uniques = df[query].shape[0], df[query].time.unique().shape[0]
        if df[query].shape[0] != df[query].time.unique().shape[0]:
            duplicates.append({'crew': crew, 'seat': seat, 'experiment': experiment, 
                                'counts': [raw_count, uniques]})
    return pd.DataFrame.from_records(duplicates)
```
* Okay so good news is indeed no duplicates are visible. 

```python
%time dupesdf = calc_duplicates(df)
# Wall time: 28.5 s

dupesdf.shape # (0, 0)
%time bumpsdf = calc_bumps(df)

```

#### So I'm going to then just sort and be done with this question
```python
%time df.sort_values(by=['crew', 'experiment', 'time'], inplace=True)
Wall time: 3.35 s

# So now if i rerun my bump func i should get empty basically..

bumpsdf.bumps.value_counts()
0    48
Name: bumps, dtype: int64

df.to_csv('data/sorted_train.csv', index=False)
```
* Le yay.


#### Reading LSTM article
* ([this](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) one)
* Also came across [link](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

#### Initial thoughts, 
* not sure if we'll need a bidirectional NN here. 
