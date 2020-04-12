
#### Summary
_Here, below, I write a bit retrospectively about my notes, trying to summarize some of the journey. I try to give some high lights
from my various notebook entries._

#### As a quick intro to the data
The physiological data includes several  types _(including respiration, electrocardiograms (ecg heart data), galvanic skin response (gsr), electroencephalography (eeg brain brain data))_ across multiple `"crews"` . A crew includes two `"seats"` (`0` and `1`). We are provided with `256 measurements per second` across three experiments (Channelized Attention (CA) , Diverted Attention (DA) and Startle/Surprise (SS) ). Across the three experiments, four target "states" (or classes) are labeled for all of the rows in the data.

The four classes of events in the training data, `'A', 'B', 'C', 'D'` , correspond to three target physiological states of the three experiments, plus a neutral baseline state:

|label|description|
|--|--|
|A | Baseline |
|B |Startle/Surprise|
|C|Channelized Attention|
|D|Diverted Attention|

In my early [notebook](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-05-10-initial-look.md) I took a quick look at the proportion of experiment time had been spent in the different states. Per the below, looking at each person separately, I saw that for the first person as an example, (`crew=1, seat=0`) , `~98%` of `CA` was labeled as "Channelized Attention" and `2%` as baseline. But for the other two experiments, the target states appear to be much more brief, with only `13%` of the "DA" and `9%` of the "SS" experiments.

_(copying an output from that notebook...)_

```
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

#### Trickiness of the how the data is laid out (crews and seats?!)
In the process of visualizing data, I had been using matplot lib to visualize the four different classes of events, `'A', 'B', 'C', 'D'` as red, green, blue and cyan. That way I could potentially try to get an intuition around the visual cues around state transitions. But at one point I had accidentally been combining the data of multiple people.

![png](2019-10-26_files/2019-10-26_14_0.png)

Above, extracting a plot from my [2019-10-26 notebook](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-10-26.md), is an example of where I plot combined multi seat data by accident. At one point this was weirding me out. But I realized finally that I had been combining the data of multiple people.

For diagram above (^^) , I had written a quick function `produce_plots_for_col` for plotting four features simultaneously, given a pandas datagrame, some features and an interval, but indeed the _zig zag_ plot was a bit baffling for a bit.

```python
start = 3400; produce_plots_for_col(df, ['r', 'ecg', 'gsr', 'eeg_fp1'],
                                range(start,start+150))
```

When I look back at my notebook I wrote about how I was very _dumbfounded_ when I realized I combined them by accident. The data is complicated however. It includes four indexing columns: `id` , `time` , `crew` and `seat` . And indeed being careful with splitting this was key in creating good datasets.

#### Some more visual inspection 
* [in my 2019-06-08 notebook](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-06-08-visually-inspect-generated-sequences.md)

#### Building datasets
* I spent a lot of time next building data sets, [here](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-06-23-today.md) , [and](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-07-06-today.md)  , and building basic quick and dirty LSTM tensor flow models. [And also](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-07-13-Five-more-data.md) .

I also tried different approaches for understanding the models I was building. Including looking at raw logits, as per the below graphic, from my [2019-07-13-Four notebook](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-07-13-Four.md#look-at-them-logits). I thought this was a cool method compared to a confusion matrix for instance is  because it shows the raw logits of each of the four classes, before the argmax voting observed in a confusion matrix is done. 

![png](2019-07-13-Four_files/2019-07-13-Four_6_0.png)




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


#### Active Learning: changing the training approach 
Somehow I came upon the idea of preferentially training on what your model is doing poorly on. So on [2020-01-19](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2020-01-19--update.md)  I modified my training loop so that I dynamically adjusted my training weights according to which class was being misclassified. The effect on the [training loss](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2020-01-19--update.md#plotting-train-loss-now-and-per-label-losses-too) was really interesting. Everything was way smoother.

Looking at a training loss plot from earlier ( such as from [2019-12-28](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-28.md#crashed-last-batch-but-thats-okay)  )

![png](2019-12-28_files/2019-12-28_12_1.png)  , it shows the batch loss is all over the place. That makes perferct sense perhaps, because each batch I have been using in stochastic gradient descent really is from all over the training data. And compared to the training loss plot for the two figures below (extracted from my [2020-01-19 notebook](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2020-01-19--update.md) ) the combined and per-class training batch losses  are way more stable looking. 
![png](2020-01-19--update_files/2020-01-19--update_10_0.png)

![png](2020-01-19--update_files/2020-01-19--update_10_2.png)  

The validation loss was still favoring that one class, but I decided to hold on to this technique and keep trying other things.

#### Full training set error
Next in [this notebook](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2020-01-19--more-test-evaluate-train.md) I wanted to better answer whether my particular test set perhaps had some very different size data compared to my training set, which was blowing up my test set error. I did not have enough data to better split apart my data at the moment actually, but instead I took a quick detour to compare my training mini batch loss curves to the full training set losses, during training. Naturally one would expect that if batch training losses improve that overall training set loss should also improve. Per the below diagram from my notebook, that was indeed the case.

![png](2020-01-19--more-test-evaluate-train_files/2020-01-19--more-test-evaluate-train_12_4.png)

#### Shuffling train/test 
After having consistently weird results with validation error, I decided to try re-building my train/test sets by doing a full random shuffle instead, in my [2020-02-01 notebook](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2020-02-01.md). Up until this point I had been using the fact that the data is divided into `crew1, crew2, crew3, etc` and I have used `crew1` for train and `crew2` for test. And I had built `scalers` from my `crew1` training data, applying them to the the `crew2` test data. 

So this time around I instead built `scalers` from `crew1` and then changed my function, `build_many_scalers_from_h5` to take `scalers` as a parameter and I kept updating them with the test data. ( My scalers `'history/2020-02-02T044441Z/scalers.joblib'` was the restulting artifact). 

In [validation](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2020-02-01.md#final-chart) , I think for the first time, I saw the validation error actually start going down ,  

![png](2020-02-01_files/2020-02-01_46_14.png) 

I took that further in [this 2020-02-08 notebook](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2020-02-08-take2--update--2.md) and ( _showing my figure again_ ) going 3 epochs instead, I got ..

![png](2020-02-08-take2--update--2_files/2020-02-08-take2--update--2_11_1.png)

So the bright side I take from this is that the validation loss is actually doing better for three out of four of the classes.

#### Reconsider that high dropout
Next in [2020-02-15 notebook](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2020-02-15.md), I decided to reduce my dropout slightly, after reading through [this post](https://machinelearningmastery.com/use-dropout-lstm-networks-time-series-forecasting/) about treating the dropout as yet another hyperparameter.  After retraining , across 2 epochs, I saw a validation loss curve, which looked better still. 


![png](2020-02-15_files/2020-02-15_10_44.png)

This indicates perhaps the context of hyperparameters being experimented with indeed matters . I think Andrew Ng's characterization of _"model babysitting"_.

https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2020-02-15.md











