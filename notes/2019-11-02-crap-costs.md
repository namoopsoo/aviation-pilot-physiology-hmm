
#### Oops I left a sagemaker notebook instance running

* per my billing dashboard, cost started going up around 10-05 to 10-07
* And looking at my cloudtrail, ... 
* I saw
```
"eventTime": "2019-10-05T18:25:19Z",
    "eventSource": "sagemaker.amazonaws.com",
    "eventName": "StartNotebookInstance",
```
* couldnt find the associated `UpdateNotebookInstance` , however, where the instance type was changed to a `ml.p2.xlarge` ... :scratching_head: 
* I vaguely remember doing that.
