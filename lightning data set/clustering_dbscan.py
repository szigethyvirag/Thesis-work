import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import os

from sklearn.cluster import DBSCAN
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
      if not os.path.exists("out_dbscan/"+Name): 
          os.makedirs("out_dbscan/"+Name)


      for eps in range(3,30,3): 
            dbscan = DBSCAN(eps=eps, algorithm='auto', n_jobs=-1).fit(Data)
            dbscan_labels = dbscan.labels_

            # initialize an axis
            fig, ax = plt.subplots(figsize=(8,6))

            # plot map on axis
            countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            countries.plot(color="lightgrey", ax=ax)

            # parse dates for plot's title
            df = pd.DataFrame(Data,columns=["longitude", "latitude"])
            
            # plot points
            df.plot(x="latitude", y="longitude", kind="scatter", color=dbscan_labels , colormap="YlOrRd", 
                    title=Name+" eps: "+str(eps), ax=ax, legend=False)

            # add grid
            ax.grid(b=True, alpha=0.5)
            plt.savefig("out_dbscan/"+Name +"/eps_"+str(eps)+".png")
            plt.close()



