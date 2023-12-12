import os
import numpy as np
import math

#load the timeseries
timeseries = []
for Name in os.listdir("timeseries"):
    if Name[:9] == "timeserie_old":
      file = open("timeseries_old/"+Name,mode='r')
      all_of_it = file.read()
      Data=np.array(all_of_it.split(), float)
      Data=np.reshape(Data,[-1,2])
      file.close()
      timeseries.append(Data)

# if there is no folder like this, it makes it (for the data of 12 mins)
if not os.path.exists("timeseries/"): 
          os.makedirs("timeseries/")

# the number of steps which are allowed to be missing
treshold = 6 

final_series = []
for serie in timeseries:
   # append everything that does not start with -inf
   if serie[0][0] != -math.inf:
      final_series.append(serie) 
   else:
      # the first point where this serie has a point
      first_known = np.where(serie != [-math.inf, -math.inf])[0][0] 
      attached = False
      for idx, fin in enumerate(final_series):
         last_known = np.where(fin != [-math.inf, -math.inf])[0][-1]
         
         if 0 < (first_known - last_known) and (first_known - last_known) < (treshold + 1) and math.dist(serie[first_known], fin[last_known]) < 30:
            final_series[idx][np.where(serie != [-math.inf, -math.inf])] = serie[np.where(serie != [-math.inf, -math.inf])]
            attached = True
            #linear interpolation:
            timestep = ((serie[first_known][0]-fin[last_known][0])/(first_known-last_known),(serie[first_known][1]-fin[last_known][1])/(first_known-last_known))
            
            for i, index in enumerate(range(last_known+1, first_known)):
               final_series[idx][index] = final_series[idx][last_known] + (i+1)*np.array(timestep)
            break
            
      if not attached:
         final_series.append(serie)

print('now its reduced to ' + str(len(final_series)))
for i, timeserie in enumerate(final_series):
  f=open("timeseries/serie" + str(i) ,"a+")
  for p in timeserie:
    f.write(str(p[0]) + " " + str(p[1]) + "\n")
  f.close()
            

      