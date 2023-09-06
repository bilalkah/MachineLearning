# -*- coding: utf-8 -*-

import cv2 
import numpy as np
import glob

pgmFile = cv2.imread("Font_1_J.pgm",-1)

target=np.array([1,0,0,0,0,0,0,
                 0,1,0,0,0,0,0,
                 0,0,1,0,0,0,0,
                 0,0,0,1,0,0,0,
                 0,0,0,0,1,0,0,
                 0,0,0,0,0,1,0,
                 0,0,0,0,0,0,1])

target=np.resize(target,(7,7))

row = len(pgmFile)
column = len(pgmFile[0])
dimen = (row-2)*(column-2)
inputLayer=np.array([0]*dimen,dtype="float64")
outputLayer=np.array([0]*7,dtype="float64")
output=np.array([0]*7)

weight=np.array([0.0]*dimen*7,dtype="float64")
weight=np.resize(weight,(dimen,7))


threshold = int(input("Threshold: "))
alpha = float(input("Learning rate: "))
epoch = int(input("Epoch: "))

for repeat in range(epoch):
    counter = 0
    for img in glob.glob("*.pgm"):
        pgmFile = cv2.imread(img,-1)
        row = len(pgmFile)
        column = len(pgmFile[0])
        k=0
        for i in range(row-2):
            for j in range(column-2):
                sum = 0.0
                for x in range(i,i+3):
                    for y in range(j,j+3):
                        sum +=pgmFile[x][y]
                sum = sum/9
                inputLayer[k]=sum
                k+=1
        for i in range(7):
            sum = 0.0
            for j in range(dimen):
                sum += inputLayer[j]*weight[j][i]
            outputLayer[i] = sum
            if(outputLayer[i]>=threshold):
                output[i]=1
            else:
                output[i]=0
            if(output[i]!=target[counter%7][i]):
                for x in range(dimen):
                    weight[x][i]+=alpha*inputLayer[x]*(target[counter%7][i]-output[i])
        counter+=1
            
def findimage(img):
    pgmFile = cv2.imread(img,-1)
    row = len(pgmFile)
    column = len(pgmFile[0])
    k=0
    for i in range(row-2):
        for j in range(column-2):
            sum = 0.0
            for x in range(i,i+3):
                for y in range(j,j+3):
                    sum +=pgmFile[x][y]
            sum = sum/9
            inputLayer[k]=sum
            k+=1
    for i in range(7):
        sum = 0.0
        for j in range(dimen):
            sum += inputLayer[j]*weight[j][i]
        outputLayer[i] = sum
        if(outputLayer[i]>=threshold):
            output[i]=1
        else:
            output[i]=0
    if((output==target[0]).all()):
        print("A")
    elif((output==target[1]).all()):
        print("B")
    elif((output==target[2]).all()):
        print("C")
    elif((output==target[3]).all()):
        print("D")
    elif((output==target[4]).all()):
        print("E")
    elif((output==target[5]).all()):
        print("J")
    elif((output==target[6]).all()):
        print("K")
    else:
        print(output)
    
    
for img in glob.glob("*.pgm"):
    findimage(img)

for img in glob.glob("test/"+"*.pgm"):
    findimage(img)
    