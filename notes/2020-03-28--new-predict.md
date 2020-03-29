


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
```





