
#### Took a detour to plot the sequences I'm currently building, as a sanity check...

```python
# Umm, and I want to look at a couple of these..
# Going to print some random plots of my sequence data here, for each category.
#
# row 1: "A" or "Baseline"; row2: "B" or "Startle/Surprise"; 
# row3: "C" or "Channelized Attention" ; 
# row4: "D"  or "Diverted Attention"

x = np.arange(0, 256, 1)
print(x.shape, y.shape)
fig = plt.figure() # x, y, )

for i,j in itertools.product(range(4), range(4)):
    ix = 1 + i*4 + j
    
    sample_row = examples[i][j]

    ax = fig.add_subplot(4, 4, ix)
    y = outdata['x_train'][sample_row][:,0]
    ax.scatter(x, y, color='lightblue', marker='.')
```

<img src="https://github.com/namoopsoo/aviation-pilot-physiology-hmm/blob/master/notes/assets/Screen%20Shot%202019-06-15%20at%2011.29.22%20AM.png"
width="607" height="383">
