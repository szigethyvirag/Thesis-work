import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import math

file = open("timeseries_05_07_2/timeserie0",mode='r')
all_of_it = file.read()
Data=np.array(all_of_it.split(), float)
Data=np.reshape(Data,[-1,2])
file.close()

fig, ax = plt.subplots(figsize=(8,6))
countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
countries.plot(color="lightgrey", ax=ax)


df = pd.DataFrame(Data,columns=["longitude", "latitude"])  #vált
df.plot(x="latitude", y="longitude", style='o-', ms=2, linewidth=1, color='red' , colormap="tab20", #kind="line"
                    title='Timeserie', ax=ax, legend=False)

ax.grid(b=True, alpha=0.5)
plt.show()
#plt.savefig("timeseries/"+Name +"/result.png")
#plt.close()