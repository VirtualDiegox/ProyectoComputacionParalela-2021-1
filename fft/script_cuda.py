#!/usr/bin/env python3

import subprocess
import csv

def Average(lst):
    return sum(lst) / len(lst)

num_hilos = [1,2,4,8,16,32,64,128]
num_blocks = [1,2,4,8,16,32]
intentos_cuda =[]
programa = "./filefftcuda"
times_cuda = []
csv_cuda = open("../datacsv/fft_cuda.csv", "w")
writer_cuda = csv.writer(csv_cuda)

titulos = ["Hilos"]
for bloques in num_blocks:
    titulos.append(str(bloques)+ " Bloques")

writer_cuda.writerow(titulos)

for hilos in num_hilos:
    times_cuda.append(hilos)
    for bloques in num_blocks:
        command = programa + " " + str(hilos) + " " + str(bloques)
        for n in range(0,5):
            output = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE)
            print(command)
            string = output.stdout.read().decode('utf-8')
            parts = string.split()
            
            time_cuda = parts[0]
            intentos_cuda.append(float(time_cuda))

        
        times_cuda.append(Average(intentos_cuda))
        intentos_cuda = []
    writer_cuda.writerow(times_cuda)
    times_cuda = []
