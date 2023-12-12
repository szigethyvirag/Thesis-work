import numpy as np
import geopy.distance
import statistics
import pickle

timeseries = np.load("series_array.npy")

# key: intersection, value: list -> index of the timeseries
connections = {} 

for i1, serie1 in enumerate(timeseries):
    for i2, serie2 in enumerate(timeseries):
        # skip if it is the same
        if i1 >= i2 : continue               
        if serie1[:,0][np.isfinite(serie1[:,0])].max() < serie2[:,0][np.isfinite(serie2[:,0])].min() or \
            serie2[:,0][np.isfinite(serie2[:,0])].max() < serie1[:,0][np.isfinite(serie1[:,0])].min() : continue
        if serie1[:,1][np.isfinite(serie1[:,1])].max() < serie2[:,1][np.isfinite(serie2[:,1])].min() or \
            serie2[:,1][np.isfinite(serie2[:,1])].max() < serie1[:,1][np.isfinite(serie1[:,1])].min() : continue
        steps_already_connected = False
        for step1 in serie1[np.isfinite(serie1[:,:])].reshape((-1,2)):
            # step1 and step2 are two coordinates
            for step2 in serie2[np.isfinite(serie2[:,:])].reshape((-1,2)):
                if len(connections) != 0:
                    for loc in connections:
                        if geopy.distance.geodesic(loc, step2).km < 50 and not (i2 in connections[loc]):
                            connections[loc].append(i2)
                            if i1 in connections[loc]: 
                                steps_already_connected = True
                            break
                if geopy.distance.geodesic(step1, step2).km < 50 and not steps_already_connected:
                    connections[(statistics.mean((step1[0],step2[0])),statistics.mean((step1[1],step2[1])))] = [i1,i2]
                    steps_already_connected = True
                if steps_already_connected:
                    break
            if steps_already_connected:
                    break

with open('connections.pkl', 'wb') as fp:
    pickle.dump(connections, fp)
    print('dictionary saved successfully to file')
        
