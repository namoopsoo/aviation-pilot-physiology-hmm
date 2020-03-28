


#### After adding preprocessing to predict..
* Here's the first form from [earlier notebook](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2020-03-07-run-test-set-snapshot6.md#indices-hmm)  

````bash
time python predict.py --test-loc-h5 data/2020-03-15T2032Z/finaltest_scaled.h5 \
        --batch-size 1024 \
        --model-loc history/2020-02-16T035758Z/epoch_001_batch_01760_model.h5 \
        --work-dir data/2020-03-15T2032Z \
        --eager    
```
* New form , points directly to my files like `data/test-crew-1_seat-0.csv` . To prepare for this, I also just threw files like `history/2020-02-02T044441Z/scalers.joblib` into a dir that I had similarly generated on AWS. Just recreated locally to make things easier.

````bash
time python predict.py --raw-test-loc data/test-crew-1_seat-0.csv \
        --batch-size 1024 \
        --model-loc history/2020-02-16T035758Z/epoch_001_batch_01760_model.h5 \
        --scalers-loc history/2020-02-02T044441Z/scalers.joblib \
        --work-dir data/2020-03-15T2032Z \
        --preprocess \
        --parallel \
        --eager    
```

