import pickle
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import random
import os

def plotting(Data, ax, num):
    color = f'#{random.randint(0, 0xFFFFFF):06x}'
    df = pd.DataFrame(Data,columns=["longitude", "latitude"]) 
    df.plot(x="latitude", y="longitude", style='o-', ms=2, linewidth=1, color=color , colormap="tab20", 
                    title="Timeseries ({})".format(num) , ax=ax, legend=False)

    plt.scatter(Data[0][1], Data[0][0])
    
timeseries = np.load("series_array.npy")

with open('connections.pkl', 'rb') as fp:
    connections = pickle.load(fp)

os.makedirs("visualizations")  

for idx, key in enumerate(connections):
    ig, ax = plt.subplots(figsize=(8,6))
    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    countries.plot(color="lightgrey", ax=ax)
    for serie in connections[key]:
        exit
        plotting(timeseries[serie], ax, len(connections[key]))
    ax.grid(b=True, alpha=0.5)
    plt.savefig("visualizations/vis_"+str(idx+1) +".png")
