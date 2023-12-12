import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import csv
import math
from datetime import datetime
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

timeseries = []

# it will be the length -1 of every timeserie at the end of timestep
timestep = -1 
today = datetime.now()

# Date to the desired string format
path_all = today.strftime("%Y_%m_%d_%H/")  
path_series = path_all+'timeseries/'

# This line creates the folder
os.makedirs(path_series)       
for FileName in os.listdir("new_Data_12_min"):

      Name=FileName[:-4]
      file = open("new_Data_12_min/"+FileName,mode='r')
      all_of_it = file.read()
      Data=np.array(all_of_it.split(), float)
      Data=np.reshape(Data,[-1,2])
      file.close()
      timestep += 1

      #if there is no folder like this, it makes it (for the data of 12 mins)
      if not os.path.exists(path_all+Name): 
          os.makedirs(path_all+Name)

      
      dbscan = DBSCAN(eps=11, algorithm='auto', n_jobs=-1).fit(Data)
      dbscan_labels = dbscan.labels_

      dict = {}
      for Data_, dbscan_labels_ in zip(Data, dbscan_labels):
        dict[tuple(Data_)] = int(dbscan_labels_)

      interesting_labels = []

      #iterates trough the labels (except -1 cause thats noise)
      for i in range(len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0) - 1): 
            if list(dbscan_labels).count(i) > 200:
                interesting_labels.append(i)

      
      f=open(path_all + Name + "/centers","a+")
      centers = []

      # searches for all the centres
      for i in interesting_labels: 
            center = [0,0]
            count = 0
            for key in dict:
                if dict[key] == i:
                     center[0] += list(key)[0]
                     center[1] += list(key)[1]
                     count = count + 1
            
            f.write(str(center[0]/count) + " " + str(center[1]/count) + '\n')
            centers.append((center[0]/count, center[1]/count))
            tmp_pair = (center[0]/count, center[1]/count) 
            
      # adding first element      
      if len(timeseries) == 0:      
            for center in centers:
                timeseries.append([center])
      # all the other elements
      else:                         
          done = False      
          for center in centers:    
            # iterates trough all the existing time series so far
            for i, timeserie in enumerate(timeseries): 
                # treshold : how close two centers should be
                if math.dist(timeserie[-1],center) < 20 and len(timeserie) < (timestep+1): 
                     # adding one centre
                     timeseries[i].append(center) 
                     done = True
                     break
            # if there is no existing time serie where we can add
            if not done: 
              tmp = [(-math.inf,-math.inf)] * timestep
              timeseries.append(tmp)
              timeseries[-1].append(center)
            # if this point is reached, either the centre is added to an existing time serie or a new one was created
              
          for i, timeserie in enumerate(timeseries):
              if len(timeserie) < (timestep+1):
                  timeseries[i].append((-math.inf,-math.inf))
              
      f.close()

      fig, ax = plt.subplots(figsize=(8,6))
      countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
      countries.plot(color="lightgrey", ax=ax)

      df = pd.DataFrame(Data,columns=["longitude", "latitude"])  #vált
      df.plot(x="latitude", y="longitude", kind="scatter", color=dbscan_labels , colormap="tab20", 
                    title=Name, ax=ax, legend=False)

      # add grid
      ax.grid(b=True, alpha=0.5)

      plt.savefig(path_all+Name +"/result.png")
      plt.close()
      

for i, timeserie in enumerate(timeseries):
  f=open(path_series+"/timeserie" + str(i+1) ,"a+")
  for p in timeserie:
    f.write(str(p[0]) + " " + str(p[1]) + "\n")
  f.close()
      


