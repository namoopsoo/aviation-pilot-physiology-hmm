


#### After adding preprocessing to predict..
* Here's the first form from [earlier notebook](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2020-03-07-run-test-set-snapshot6.md#indices-hmm)  

```bash
time python predict.py --test-loc-h5 data/2020-03-15T2032Z/finaltest_scaled.h5 \
        --batch-size 1024 \
        --model-loc history/2020-02-16T035758Z/epoch_001_batch_01760_model.h5 \
        --work-dir data/2020-03-15T2032Z \
        --parallel \
        --eager    
```
* New form , points directly to my files like `data/test-crew-1_seat-0.csv` . To prepare for this, I also just threw files like `history/2020-02-02T044441Z/scalers.joblib` into a dir that I had similarly generated on AWS. Just recreated locally to make things easier.

```bash
time python predict.py --raw-test-loc data/test-crew-1_seat-0.csv \
        --batch-size 1024 \
        --model-loc history/2020-02-16T035758Z/epoch_001_batch_01760_model.h5 \
        --scalers-loc history/2020-02-02T044441Z/scalers.joblib \
        --work-dir data/2020-03-15T2032Z \
        --preprocess \
        --parallel \
        --eager    
```
* Cool so using joblib parallel, this took ... not that long

```
[Parallel(n_jobs=4)]: Done 105 out of 105 | elapsed:  3.8min finished
... 
also changed to using -1 workers...
[Parallel(n_jobs=-1)]: Done  95 out of  95 | elapsed:  3.0min finished
, with the whole thing:
real	5m20.408s
user	25m30.651s
sys	3m7.183s
...
real	4m4.969s
user	18m55.300s
sys	2m19.742s
```
* so first one ... , was `test-crew-1_seat-0`  
* ok next one...

```bash
time python predict.py --raw-test-loc data/test-crew-1_seat-1.csv \
        --batch-size 1024 \
        --model-loc history/2020-02-16T035758Z/epoch_001_batch_01760_model.h5 \
        --scalers-loc history/2020-02-02T044441Z/scalers.joblib \
        --work-dir data/2020-03-15T2032Z \
        --preprocess \
        --parallel \
        --eager  
```

* NEXT : 

```
time python predict.py --raw-test-loc data/test-crew-13_seat-1.csv \
        --batch-size 1024 \
        --model-loc history/2020-02-16T035758Z/epoch_001_batch_01760_model.h5 \
        --scalers-loc history/2020-02-02T044441Z/scalers.joblib \
        --work-dir data/2020-03-15T2032Z \
        --preprocess \
        --parallel \
        --eager  
```
* ...
```


```

#### look for gaps... fill them..
```python
files = [f'{workdir}/{x}' for x in os.listdir(workdir)
        if 'crewseat-preds-test' in x]
dfs = {
        x.split('/')[-1].split('.')[0].split('preds')[1]: 
        pd.read_csv(x, index_col=None)
                        for x in files}

# At least checking for overlaps ...
In [103]: list(dfs.keys())[:3]                                                                                                     
Out[103]: ['-test-crew-13_seat-0', '-test-crew-5_seat-0', '-test-crew-1_seat-0']

In [104]: len(set(dfs['-test-crew-1_seat-1'].id.tolist()) & set(dfs['-test-crew-1_seat-0'].id.tolist()))                           
Out[104]: 0

predsdf = pd.concat([pd.read_csv(x, index_col=None)
                        for x in files])

# hmm oddly lot more missing than I expected  
In [96]: predsdf.shape                                                                                                             
Out[96]: (16452699, 5)
MAX = 17965143
set(list(range(MAX))) - set(predsdf.id.tolist())

In [108]: len(set(list(range(MAX))) - set(predsdf.id.tolist()))                                                                    
Out[108]: 1513961

fulldf = pd.DataFrame({'id': list(range(MAX))})
senddf = fulldf.merge(predsdf, how='left', on='id')
```
```python
In [112]: len(fulldf.id.unique())                                                                                                  
Out[112]: 17965143

In [113]: len(senddf.id.unique())                                                                                                  
Out[113]: 17965143

In [114]: len(predsdf.id.unique())                                                                                                 
Out[114]: 16451183

In [115]: predsdf.shape                                                                                                            
Out[115]: (16452699, 5)

In [116]: 16452699 - 16451183                                                                                                      
Out[116]: 1516

predsdf.groupby(by='id').size().reset_index()
countsdf = predsdf.groupby(by='id').size().reset_index().rename(columns={0: 'count'})
countsdf[countsdf['count'] > 1].shape
# Out[121]: (1515, 2)

countsdf[countsdf['count'] > 1].merge(predsdf,  how='left', on='id')

In [122]: countsdf[countsdf['count'] > 1].merge(predsdf,  how='left', on='id').shape                                               
Out[122]: (3031, 6)

# ok looked at the dupes and they are all the same somehow. 
In [123]: countsdf[countsdf['count'] > 1].merge(predsdf,  how='left', on='id').to_csv('data/2020-03-15T2032Z/dupesdf.csv')         
```
* ok then look at gaps ..
```python
senddf.drop_duplicates(subset='id')

In [125]: senddf.drop_duplicates(subset='id').shape, senddf.shape                                                                  
Out[125]: ((17965143, 5), (17966659, 5))


In [126]: senddf.drop_duplicates(subset='id').to_csv('data/2020-03-15T2032Z/senddf.csv')                                           

# observing some of the gaps...

In [130]: senddf[senddf['0'].isnull()].shape                                                                                       
Out[130]: (1513964, 5)


```
* VIsual inspection of the nulls... 
```
580861
779168

nulls = senddf[senddf['0'].isnull()].id.tolist()

In [144]: senddf.drop_duplicates(subset='id').id.tolist() == fulldf.id.tolist()                                                    
Out[144]: True
```
* not sure why there are nulls like this ...
```python
In [147]: senddf[senddf.id > 580851].iloc[:10]                                                                                     
Out[147]: 
            id    0    1    2    3
580905  580852  2.5 -1.9 -2.2  1.8
580906  580853  NaN  NaN  NaN  NaN
580907  580854  2.5 -1.9 -2.2  1.8
580908  580855  NaN  NaN  NaN  NaN
580909  580856  2.5 -1.9 -2.2  1.8
580910  580857  NaN  NaN  NaN  NaN
580911  580858  2.5 -1.9 -2.2  1.8
580912  580859  NaN  NaN  NaN  NaN
580913  580860  2.5 -1.9 -2.2  1.8
580914  580861  NaN  NaN  NaN  NaN
```
* But I grepped my original data files and that id is not missing...
```
$ egrep '^580853' data/test-crew*
data/test-crew-1_seat-1.csv:580853,1,LOFT,2018.47656250,1,0.0,0.0,-0.000488,-0.000244,0.0,0.000244,0.000244,0.000244,0.000244,0.000244,1.10181,-0.000244,0.016602000000000002,0.0,0.000244,0.0,0.000122,0.0,0.001221,2958.639893,-6261.25,827.987976,0.0
```
* Ok, is it in the output prediction df?

```
580853 dfs['-test-crew-1_seat-1']

In [150]: dfs['-test-crew-1_seat-1'].id.tolist()[:4]                                                                               
Out[150]: [1044255, 1044257, 1044259, 1044261]

In [151]: 1044259 in dfs['-test-crew-1_seat-1'].id.tolist()[:4]                                                                    
Out[151]: True

In [152]: 580853 in dfs['-test-crew-1_seat-1'].id.tolist()[:4]                                                                     
Out[152]: False


```
```
$ wc data/test-crew-1_seat-1.csv
 1037808 1037808 288736415 data/test-crew-1_seat-1.csv
```
```
In [155]: dfs['-test-crew-1_seat-1'].shape, dfs['-test-crew-1_seat-1'].id.unique().shape                                           
Out[155]: ((978226, 5), (978134,))
```
* umm, `59674 = 1037808 - 978134`  , the number missing may be related to my batching..

* This is pretty weird though... might help explain things. Sorting by time is not sorting by id... so what the faack? 
```
In [156]: df11 = pd.read_csv('data/test-crew-1_seat-1.csv')                                                                        

In [158]: df11.sort_values(by='id').id.tolist() == df11.sort_values(by='time').id.tolist()                                         
Out[158]: False


```
* ok just fill it based on the way files are laid naturally.. with `backfill` 
```
files = [f'{workdir}/{x}' for x in os.listdir(workdir)
        if 'crewseat-preds-test' in x]

predsdf = pd.concat([pd.read_csv(x, index_col=None)
                        for x in files]
          ).drop_duplicates(subset='id')

fulldf = pd.DataFrame({'id': list(range(MAX))})
senddf = fulldf.merge(predsdf, how='left', on='id'
            ).fillna(method='backfill'
            ).fillna(method='ffill'
            ).sort_values(by='id'
            ).rename(columns={'0': 'A', '1': 'B', '2': 'C', '3': 'D'})
            
for k in ['A', 'B', 'C', 'D']:
    senddf[k] = senddf[k].map(lambda x: max(0, x))
    
senddf['argmax'] = pd.DataFrame(senddf.iloc[9:11].apply(

                    np.array(lambda x: [int(x.id)] + {0: [1, 0, 0, 0],
                             1: [0, 1, 0, 0],
                             2: [0, 0, 1, 0],
                             3: [0, 0, 0, 1],}.get(np.argmax([x.A, x.B, x.C, x.D]))), axis=1),
                             columns=['id', 'A', 'B', 'C', 'D'])
#
normsenddf = senddf.apply(
                   lambda x: {
                           0: pd.Series({'id': int(x.id), 'A': 1, 'B': 0, 'C': 0, 'D': 0}),
                           1: pd.Series({'id': int(x.id), 'A': 0, 'B': 1, 'C': 0, 'D': 0}),
                           2: pd.Series({'id': int(x.id), 'A': 0, 'B': 0, 'C': 1, 'D': 0}),
                           3: pd.Series({'id': int(x.id), 'A': 0, 'B': 0, 'C': 0, 'D': 1}),
                   }.get(np.argmax([x.A, x.B, x.C, x.D])), axis=1)


    
senddf.to_csv(f'{workdir}/{mu.getts()}-sendit.csv', index=False)


```
* I had initially only used the `.fillna(method='backfill')` which meant the very last row was still null,
* I submitted that to Kaggle, but got an error about it. Since by definition the last row has no leading row to backfill from.
* So I added a `ffill` forward fill as well.. going to resubmit

```
17965141,1.7,-0.8,-2.0,1.2
17965142,-0.0,0.1,-0.2,0.0
17965143,,,,
```
* ok now i have 

```
id,A,B,C,D
0,-0.3,-0.1,-0.0,-0.1
1,-0.3,-0.1,-0.0,-0.1
2,-0.3,-0.1,-0.0,-0.1
3,-0.3,-0.1,-0.0,-0.1
...
17965140,-0.0,0.1,-0.2,0.0
17965141,1.7,-0.8,-2.0,1.2
17965142,-0.0,0.1,-0.2,0.0
```
* 
```
for k in ['A', 'B', 'C', 'D']:
    senddf[k] = senddf[k].map(lambda x: max(0, x))
    
senddf['argmax'] = pd.DataFrame(senddf.iloc[9:11].apply(

                    np.array(lambda x: [int(x.id)] + {0: [1, 0, 0, 0],
                             1: [0, 1, 0, 0],
                             2: [0, 0, 1, 0],
                             3: [0, 0, 0, 1],}.get(np.argmax([x.A, x.B, x.C, x.D]))), axis=1),
                             columns=['id', 'A', 'B', 'C', 'D'])
```

#### ... fuller 
```python
import pandas as pd

workdir = 'data/2020-03-15T2032Z'   
files = [f'{workdir}/{x}' for x in os.listdir(workdir)
        if 'crewseat-preds-test' in x]

predsdf = pd.concat([pd.read_csv(x, index_col=None)
                        for x in files]
          ).drop_duplicates(subset='id')
MAX = 17965143

fulldf = pd.DataFrame({'id': list(range(MAX))})
senddf = fulldf.merge(predsdf, how='left', on='id'
            ).fillna(method='backfill'
            ).fillna(method='ffill'
            ).sort_values(by='id'
            ).rename(columns={'0': 'A', '1': 'B', '2': 'C', '3': 'D'})

%%time
senddf['argmax'] = senddf.apply(lambda x: np.argmax([x.A, x.B, x.C, x.D]), axis=1)
#
In [12]: %%time 
    ...: senddf['argmax'] = senddf.apply(lambda x: np.argmax([x.A, x.B, x.C, x.D]), axis=1) 
    ...: #                                                                                                                      
CPU times: user 24min 16s, sys: 8.07 s, total: 24min 24s
Wall time: 24min 33s

In [13]: senddf['argmax'].value_counts()                                                                                           
Out[13]: 
0    13292040
1     3836332
2      688643
3      148128
Name: argmax, dtype: int64




# 

normsenddf = senddf.apply(
                   lambda x: {
                           0: pd.Series({'id': int(x.id), 'A': 1, 'B': 0, 'C': 0, 'D': 0}),
                           1: pd.Series({'id': int(x.id), 'A': 0, 'B': 1, 'C': 0, 'D': 0}),
                           2: pd.Series({'id': int(x.id), 'A': 0, 'B': 0, 'C': 1, 'D': 0}),
                           3: pd.Series({'id': int(x.id), 'A': 0, 'B': 0, 'C': 0, 'D': 1}),
                   }.get(np.argmax([x.A, x.B, x.C, x.D])), axis=1)
    
normsenddf.to_csv(f'{workdir}/{mu.getts()}-sendit.csv', index=False)
```



* One file at atime....
```
import os
import pandas as pd
from tqdm import tqdm
workdir = 'data/2020-03-15T2032Z'   
files = [f'{workdir}/{x}' for x in os.listdir(workdir)
        if 'crewseat-preds-test' in x]
for file in tqdm(files):
    df = pd.read_csv(file, index_col=None
        ).rename(columns={'0': 'A', '1': 'B', '2': 'C', '3': 'D'})

    newfile = f'{workdir}/normalized/normed-{file.split("/")[-1]}'
    newdf = df.apply(
                   lambda x: {
                           0: pd.Series({'id': int(x.id), 'A': 1, 'B': 0, 'C': 0, 'D': 0}),
                           1: pd.Series({'id': int(x.id), 'A': 0, 'B': 1, 'C': 0, 'D': 0}),
                           2: pd.Series({'id': int(x.id), 'A': 0, 'B': 0, 'C': 1, 'D': 0}),
                           3: pd.Series({'id': int(x.id), 'A': 0, 'B': 0, 'C': 0, 'D': 1}),
                   }.get(np.argmax([x.A, x.B, x.C, x.D])), axis=1)
    newdf.to_csv()
predsdf = pd.concat([pd.read_csv(x, index_col=None)
                        for x in files]
          ).drop_duplicates(subset='id')


```

* 18:22 ... trying just small one for now . ^^


