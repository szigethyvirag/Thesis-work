import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import os
from sklearn.cluster import KMeans
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

      # if there is no folder like this, it makes it (for the data of 12 mins)
      if not os.path.exists("out_kmeans"+Name): 
          os.makedirs("out_kmeans/"+Name)


      for ClusterNum in range(2,20):
            kmeans = KMeans(n_clusters=ClusterNum, random_state=0, n_init="auto").fit(Data)
            kmeans_labels = kmeans.labels_

            # initialize an axis
            fig, ax = plt.subplots(figsize=(8,6))

            # plot map on axis
            countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            countries.plot(color="lightgrey", ax=ax)

            # parse dates for plot's title
            df = pd.DataFrame(Data,columns=["longitude", "latitude"])

            df.plot(x="latitude", y="longitude", kind="scatter", color=kmeans_labels , colormap="gist_rainbow", 
                    title=Name+" ClusterNum: "+str(ClusterNum) , ax=ax, legend=False)

            # add grid
            ax.grid(b=True, alpha=0.5)
            plt.savefig("out_kmeans/"+Name +"/clusters_"+str(ClusterNum)+".png")
            plt.close()




