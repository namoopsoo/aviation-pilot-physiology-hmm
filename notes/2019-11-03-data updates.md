

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
# True
```
* If I try writing to that again... with `r+` .. what will happen? 
```python
with h5py.File("data.h5", "r+") as f:
    print(f.keys())    
# <KeysViewHDF5 ['dataset_1']>

vec1 = np.random.random(size=(100,20))
with h5py.File("data.h5", "r+") as f:
  f.create_dataset('dataset_2', data=vec1)
  print(f.keys())    
# <KeysViewHDF5 ['dataset_1', 'dataset_2']>

```
* Ah nice so now theres `2` datasets. ^^ 
```python
with h5py.File("data.h5", "r+") as f:
    print(f['dataset_1'][:2,:5])
    print(f['dataset_2'][:2,:5])
# =>
[[0.01069223 0.77314218 0.04700203 0.79063396 0.17771211]
 [0.19120064 0.93442174 0.48662097 0.41833657 0.1051533 ]]
[[0.89912131 0.78146439 0.52470407 0.91182476 0.84383107]
 [0.36944307 0.69090266 0.55056888 0.4260522  0.62340194]]
```

* I want to try this _append-looking_ file mode, `a`, although it is described as _"Read/write if exists, create otherwise"_ not _"append"_.  Ah , actually `r+` says `Read/write, file must exist` while `a` says `Read/write if exists, create otherwise`
* So lets see, using `r+` on a new file name should fail?
```python
try:
    with h5py.File("datanewnew.h5", "r+") as f:
        print(f.keys())
except Exception as e:
    print(e)

# =>
# Unable to open file (unable to open file: name = 'datanewnew.h5', errno = 2, error message = 'No such file or directory', flags = 1, o_flags = 2)
```
* And using `a` ?
```python
with h5py.File("datanewnew.h5", "a") as f:
    print(f.keys())
    
# Agh indeed.. ==> no exception
# <KeysViewHDF5 []>
# just empty 
# and it did create a file there. 

```

#### Also, no strings to h5
* Sort of obvious, but when i tried , this failed..
```python
with h5py.File("datanewnew.h5", "a") as f: 
    f.create_dataset('dataset_foo', data={'ok': vec1})

TypeError: Object dtype dtype('O') has no native HDF5 equivalent
```
* Actually less ovvious, same error I got when simply trying to save a python list. Like a list of numpy arrays.
