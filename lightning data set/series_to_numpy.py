import os
import numpy as np

#load the timeseries

# rename it to the set of series you want to work with
measurement = "2023_10_27_23"      
path = measurement + "/timeseries"
for Name in os.listdir(path):
    #Name=FileName[:-4]
    if Name[:9] == "timeserie":
      file = open(path+"/"+Name,mode='r')
      all_of_it = file.read()
      Data=np.array(all_of_it.split(), float)
      Data=np.reshape(Data,[-1,2])
      file.close()
      if Name == "timeserie1":
        timeseries = np.array([Data])
      else:
        timeseries = np.append(timeseries, [Data], axis=0)

np.save(measurement+"/series_array.npy", timeseries)