### Aviation-pilot-physiology-hmm
A look at this physiological data kaggle from [earlier this year](https://www.kaggle.com/c/reducing-commercial-aviation-fatalities/data).

### Notes
- [My High Level Recap](#my-high-level-recap)
- [_Initial look_](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-05-10-initial-look.md)
- [_Test driving TensorFlow with the fashion hello world_](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-05-13--keras-hello-world-fashion.ipynb)
- [_Wrangling time data_](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-05-14-wrangling-time-data.md)
- [_LSTM dive_](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-05-15--lstm%3F.md)
- [More](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-05-17--what%3F.md)
- [The losses](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-05-18.md)
- [Visual inspection detour](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-06-08-visually-inspect-generated-sequences.md)
- [Trying with eager execution next](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-06-09--hmm-try-eager.md)
- [Changing up the weights](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-06-15--more-eager.md)
- [More data, less stratifying](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-06-16--today.md.ipynb)

- [Test driving TensorBoard](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-08-10-c-again.ipynb), which required a much different train loop that stores ouputs from the logits across the training epochs. Very cool.
- [Using h5py](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-01-batch-train.md) for data now [and plotting my logloss per class](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-08-.md) .
- [Really nice MinMaxScaler partial fit approach](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-21--update.md)


### My High Level Recap
This physiology data classification challenge poses the question, _given this time series voltage data of pilots' respiration, electrocardiograms (ecg heart data), galvanic skin response (gsr), electroencephalography (eeg brain brain [data](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2020-04-05-organzing-thoughts.md#as-a-quick-intro-to-the-data))  can something reliable be said about their physiological state?_ My response was to use this as an opportunity to learn about TensorFlow and **LSTMs**. I quickly discovered that data processing around time series data is `3 dimensional` as opposed to typical `2 dimensional` data. That means that the harmless `1.1 GiB` of training data can quickly multiply to roughly `256 GiB` if one is interested in using a `256 long` sequence window. That means I learned a lot more about `numpy` for its simplicity around transforming `3 dimensional` data.  I had to adapt to using `h5py` *chunks* of data so as not to run out of memory quickly and not wait endless hours for training sessions to merely crash. As for  *TensorFlow* and *LSTMs*, I did not realize right away but *LSTMs* (and likely neural nets in general) are quite sensitive to data that is not scaled and my logloss ended up reducing when I [applied scaling techniques](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2020-04-05-organzing-thoughts.md#scaling). Raw `matplotlib` became more intuitive for helping to visualize not just the time series data itself, but also for plotting  `logloss` while training, across *batches* . . My **Jupyter Notebook** hygiene and workflow got better really quickly too, because I needed a reliable tool for distinguishing one day's experiment from another's without mixing the  together the data or the models.

This dataset was highly skewed in terms of *classes* and so one of the really important preprocessing tasks was creating both *balanced* training "techniques" and *balanced* test data for more uniformly judging model performance. And I say "techniques", because I tried both balanced weights and balanced training sets. I ended up preferring balanced training datasets, because that meant less preprocessing code.

The picture of *Kaggle* data I had in my head was clean datasets, but this dataset had one huge problem in that the time series data was not actually sorted by, you know, `time`. But in a way it is always fun to deal with messy data because it makes you more engaged with it and still more curious in the outcomes.

*In general this project has given me a lot of fun memories.*

One day after already getting deep into my **LSTM** approach, I decided to look through the *Kaggle* discussions for this project and I found that most people actually stuck to gradient boosting machines like lightGBM or XGBoost. But I decided to follow my personal motto of taking the path less traveled so I kept going with the *LSTM* .

I have spent I think half a year of weekends on this problem. I have memories of learning about neural network architecture learning "capacity"  at my niece's birthday party. I came to understand that creating a larger network can cause it to *memorize* more as opposed to *generalize* .

I remember tweaking my *stochastic gradient descent* batch size after reading this Yann LeCun [tweet](https://mobile.twitter.com/ylecun/status/989610208497360896) , *"Training with large minibatches is bad for your health. More importantly, it's bad for your test error. Friends dont let friends use minibatches larger than 32."*  .

I also have memories of starting modeling experiments before going on runs and before going to sleep, so that I could let *TensorFlow* spin its wheels while I took my mind into strategy mode or just let myself meditate.

At one point I was at the Boston Amtrak terminal waiting for my bus, getting deeper into why it is handy to look at raw *logit* data coming out of a model, especially in a multiclass problem because it can show how strongly a model classifies each class. But applying the logistic function or a *softmax* is of course good for sussing out probabilities. But then I realized I was waiting for a bus at an Amtrak terminal and I had to sprint several blocks to actually catch my bus!

At the end of the day I think of all of the amazing things I could one day do with this kind of technology, such as classifying music or building a chat bot (maybe even one that can tell jokes).


### Pocket glossary
* Jupyter notebooks are to data science(s) what paper bound lab notebooks are to physical sciences (physics? chemistry?).
* Kaggle.com is a website where you can compete at solving data science problems with a model that has the lowest loss.
* LSTMs or Long Short Term Memory "networks" are a type of recurrent neural networks which are better with patterns which require more information from the past.
* matplotlib is a software package for plotting data.
* numpy is a software package for more easily performing linear algebra transformations and asking statistical questions of your data.
* Recurrent Neural Networks (RNNs) are neural networks which can learn "sequences" of data.
* TensorFlow is a software package which provides implementations of neural net algorithms for solving image classification, natural language and other similar problems which classical machine learning algorithms like logistic regression struggle with.

