#!/usr/bin/env python3

import subprocess
import csv

def Average(lst):
    return sum(lst) / len(lst)

num_hilos = [1,2,4,6,8,10,12,16]
intentos_omp =[]
programa = "./filefftomp"
times_omp = []
csv_omp = open("../datacsv/fft_omp.csv", "w")
writer_omp = csv.writer(csv_omp)


for hilos in num_hilos:
    command = programa + " " + str(hilos)

    for n in range(0,5):
        output = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE)
        print(command)
        string = output.stdout.read().decode('utf-8')
        parts = string.split()
        
           
        time_omp = parts[0]
        intentos_omp.append(float(time_omp))

    times_omp.append(hilos)
    times_omp.append(Average(intentos_omp))
    intentos_omp = []
    writer_omp.writerow(times_omp)
    times_omp = []
