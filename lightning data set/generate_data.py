import os
import sys

""" 
This file stands for making 12 min cuts from our raw data
"""

def choose_a_time(time):
    hours = time[:2]
    if int(time[2:]) < 12:
        minutes = '00'
    elif int(time[2:]) < 24:
        minutes = '12'
    elif int(time[2:]) < 36:
        minutes = '24'
    elif int(time[2:]) < 48:
        minutes = '36'
    elif int(time[2:]) < 60:
        minutes = '48'
    else:
        sys.stderr.write("Something went really wrong with the minutes :( \n")
        exit()
    return hours+ '_' + minutes




os.makedirs("new_Data_12_min")
for FileName in os.listdir("new_Data"):
    f = open('new_Data/ENTLN_20190114_info.txt', 'r')
    date = "20190114"
    
    for line in f:
        line = line.split(' ')
        time = line[0][11:16].replace(':', '')
        time_12 = choose_a_time(time)

        with open("new_Data_12_min/"+date+"_"+time_12+".txt", "a+") as txt_file:
            txt_file.write(line[1] + " " + line[2] + "\n")
            txt_file.close()
    break
