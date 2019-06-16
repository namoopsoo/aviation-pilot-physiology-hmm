_Draft.. still need to copy from notebook..._

#### At one point I did this... 
```python
target_indices = examples[0] + examples[1] + examples[2] + examples[3]
print(len(target_indices))

%time examples = {j: fetch_some_examples(outdata, j, n=1000) for j in range(4)}

%time dataset_batches = build_dataset(outdata, target_indices)

print(dataset_batches)
```
