

#### Hmm h5py recipies ...

* tensor flow models get saved as `.h5` . Looking through the [h5py](http://docs.h5py.org/en/latest/quick.html) docs, I also see this `.hdf5` format.
* One recipe I saw athat got bunch of upvotes on stack overflow..

```python
# Write
import numpy as np
import h5py
a = np.random.random(size=(100,20))
h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('dataset_1', data=a)
# <HDF5 dataset "dataset_1": shape (100, 20), type "<f8">
h5f.close()

# Read
h5f = h5py.File('data.h5','r')
b = h5f['dataset_1'][:]
h5f.close()

np.allclose(a,b)
True
```
