
### Scaling

[On 2019-12-21](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-21--update.md) , I ended up trying out more scaling approaches, especially `MinMaxScaler`. I had `8` features I was focusing on at that point and I plotted how my `minMaxScaler` `min` and `max` parameters changed as I processed roughly `40` or so mini datasets I had in my h5 training file `data/2019-12-21T215926Z/train.h5`. _Re-posting [my image](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-21--update.md#plot-the-scale-parameters)_  : 

![png](2019-12-21--update_files/2019-12-21--update_17_0.png)

Luckily I found I was able to use just a single sklearn MinMaxScaler object to capture all `8` features at once. 
I then  [applied](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-21--update.md#make-scaled-data) the scalers to transform my `train.h5` data to a  `train_scaled.h5` dataset. And I also ended up with [balanced datasets](https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/2019-12-21--update.md#ok-now-make-balanced-data-again) that I could use for training.

