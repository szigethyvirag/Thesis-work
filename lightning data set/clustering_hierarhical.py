import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import os

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

for FileName in os.listdir("Data_lightning"):

      Name=FileName[:-4]


      file = open("Data_lightning/"+FileName,mode='r')
       
      # read all lines at once
      all_of_it = file.read()
      Data=np.array(all_of_it.split(), float)
      Data=np.reshape(Data,[-1,2])
      
      # close the file
      file.close()


      #which kind of linkage we want to use during making the dendogram 
      linked = linkage(Data, 'single') 

      labelList = range(1, Data.shape[0]+10) #??

      #if there is no folder like this, it makes it 
      if not os.path.exists("out_hierarchical/"+Name): 
          os.makedirs("out_hierarchical/"+Name)


      for ClusterNum in range(2,20):
            # forms flat clusters from the hierarchical clustering
            fl = fcluster(linked,ClusterNum,criterion='inconsistent') 

            # initialize an axis
            fig, ax = plt.subplots(figsize=(8,6))

            # plot map on axis
            countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            countries.plot(color="lightgrey", ax=ax)

            # parse dates for plot's title
            df = pd.DataFrame(Data,columns=["longitude", "latitude"])

            # plot points
            df.plot(x="latitude", y="longitude", kind="scatter", color=fl , colormap="YlOrRd", 
                    title=Name+" ClusterNum: "+str(ClusterNum) , ax=ax, legend=False)

            # add grid
            ax.grid(b=True, alpha=0.5)
            plt.savefig("out_hierarchical/"+Name +"/clusters_"+str(ClusterNum)+".png")
            plt.close()

