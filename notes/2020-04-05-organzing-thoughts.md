Here I write a bit retrospectively about my notes, trying to summarize some of the journey.

### Scaling
I took a deeper [histogram](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-14--annotated.md) look at my data, seeing quite a lot of [ups and downs](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-14--annotated.md#another-time-series-look). 

_(Given that there were some crazy jumps, I thought I needed to do something about that)_
![png](2019-12-14--annotated_files/2019-12-14--annotated_16_0.png)


And so 
[on 2019-12-21](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-21--update.md) , I ended up trying out more scaling approaches, especially `MinMaxScaler`. I had `8` features I was focusing on at that point and I plotted how my `minMaxScaler` `min` and `max` parameters changed as I processed roughly `40` or so mini datasets I had in my h5 training file `data/2019-12-21T215926Z/train.h5`. _Re-posting [my image](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-21--update.md#plot-the-scale-parameters)_  : 

![png](2019-12-21--update_files/2019-12-21--update_17_0.png)

Luckily I found I was able to use just a single sklearn MinMaxScaler object to capture all `8` features at once. 
I then  [applied](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-21--update.md#make-scaled-data) the scalers to transform my `train.h5` data to a  `train_scaled.h5` dataset. And I also ended up with a [balanced dataset](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-21--update.md#ok-now-make-balanced-data-again) , `train_balanced.h5`, that I could use for training.

I trained a model and plotted training and validation loss curves the [next day](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-22.md) . 

And wow the validation loss ( [link](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-22.md#plotting-validation-loss-at-model-snapshots) ) looked intense , 

![png](2019-12-22_files/2019-12-22_11_1.png)

As a side note. although the validation loss here looks totally skewed towards `class 1` , I want to step back and note I really appreciate the technique of actually creating the _"balanced"_ test set I referred to above. That allows us to quickly knows the model is favoring one class over another in the first place. And also I really dig the technique of simply snapshotting the tensorflow models while training and then being able to know how the validation logloss looks across those training batches. I feel like combining these techniques was really helpful in digesting what is going on . I needed to enjoy little details like that amidst all of the trial and error that was happening here (Emphasis on the error part haha).


#### Shuffling and adjusting dropout
At a [later date](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-25.md) , I adjusted my  lstm dropout from `0.2` to `0.7` , seeing quite different behavior in the validation loss. I had also added some [shuffling code](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-25.md#do-a-shuffle) taking my `'history/2019-12-22T174803Z/train_balanced.h5'` dataset to produce `'history/2019-12-22T174803Z/train_scaled_balanced_shuffled.h5'` , to possibly change some of the choppiness of the validation curve seen above ^^ . That produced a [validation loss](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-25.md#validation-loss) , reposting the image here, 

![png](2019-12-25_files/2019-12-25_13_1.png)



#### More epochs?
On [2018-12-28](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-28-two.md) I got curious about whether just  throwing more data at this problem would help. So I extended my waiting time by `two` and let the training happen in two *epochs* . The validation loss [from here](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-28-two-plot.md#final-validation-logloss-plot)  , (reposting...) however showed that throwing more data is not always the answer. It always depends haha.

![png](2019-12-28-two-plot_files/2019-12-28-two-plot_9_1.png)


#### Weight initialization
Per my [notebook entry](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2020-01-12.md) I had read per [this article](https://adventuresinmachinelearning.com/weight-initialization-tutorial-tensorflow/) that the default tensor flow weight initialization I had [been using](https://www.tensorflow.org/api_docs/python/tf/keras/initializers) was  _GlorotUniform , ( which is aka Xavier Uniform apparently )_ . I realized it was at least worth considering weight initialization as another hyper parameter so here I tried the  _Glorot or Xavier Normal_ instead .  The [validation loss](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2020-01-12.md#validation-loss) did not necessarily convey the difference however: 

![png](2020-01-12_files/2020-01-12_16_48.png)

At this point I think I was realizing that the order of ideas to try matters. And you do not know in advance what is the best order. Perhaps the weight initialization matters a good deal, but I had not yet found the critical next step yet at that point.

#### Class balance
In my [next notebook](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2020-01-18.md) I wanted to understand why my `class 1` kept getting favored. I tried out [forcing the weights](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2020-01-18.md#force-weights) of my training data to basically 

```
{0: 1., 1: 0., 2: 0., 3: 0.}
```

to see what happens and sure enough, per the [validation loss](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2020-01-18.md#validation-loss) , the loss now went down only for class `0`. So the effect was controlled. 

![png](2020-01-18_files/2020-01-18_11_4.png)



