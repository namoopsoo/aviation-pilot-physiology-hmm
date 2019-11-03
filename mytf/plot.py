import numpy as np

def make_series_from_cols(df, col, indices):
    label_col = 'event'    
    return [[
        x if y == event else np.nan
     
        for (x, y)
        in df[[col, label_col]].loc[indices].values]
        
        for event in ['A', 'B', 'C', 'D']
    ] 


def produce_plots_for_col(df, cols, indices):
    event_col = 'event'
    # plot for each event though..
    fig = plt.figure(figsize=(12,8))

    assert len(cols) == 4
    
    for i, col in enumerate(cols):
        Y = make_series_from_cols(df, col, indices)
        ax = fig.add_subplot(int('41' + str(i+1)))
        
        ax.plot(df.loc[indices].time, Y[0], 'r+-')
        ax.plot(df.loc[indices].time, Y[1], 'g+-')
        ax.plot(df.loc[indices].time, Y[2], 'b+-')
        ax.plot(df.loc[indices].time, Y[3], 'c+-')
        
        ax.set(#title=col,
            ylabel=col,
            xlabel='time')

