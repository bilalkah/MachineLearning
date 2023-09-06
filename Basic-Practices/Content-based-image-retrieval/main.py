# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 19:52:25 2020

@author: bilal
"""

from PIL import Image
from numpy import asarray, array
import math
import glob

def findWhichRGB(str):
    """
    Bu fonskiyon ilk olarak parametre olarak girilen resmin RGB histogramını çıkarırır
    daha sonra çıkarılan RGB historamıyla databasede bulunan RGB histogramlar arasındaki
    Öklid mesafesini ölçer. En yakın 5 resme ait bilgileri bir matriste kaydeder ve bu matrisi döndürür
    """
    #Burada resmi 3 boyutlu bir matris olarak data'ya alırız
    image = Image.open(str)
    data = asarray(image)
    
    #Satır ve sütun sayılarını row ve column'a alırız
    row = len(data)
    column = len(data[0])
    
    #256 birimlik rgb histogramlarına başlangıç olarak 0 atarız
    histR = array([0]*256)
    histG = array([0]*256)
    histB = array([0]*256)
    
    #data[i][j][0]=R, data[i][j][0]=G, data[i][j][0]=B olacak şekilde histogramını alırız
    for i in range(row):
        for j in range(column):
            histR[data[i][j][0]] += 1
            histG[data[i][j][1]] += 1
            histB[data[i][j][2]] += 1
            
    #Resimlerin boyutları birbirlerinden farklı olacağı için bunları aynı hizaya çekeriz
    for i in range(256):
        histR[i] = 8192*histR[i]/(row*column)
        histG[i] = 8192*histG[i]/(row*column)
        histB[i] = 8192*histB[i]/(row*column)
    
    #Database e erişmek için database.txt dosyasını reading modda açarız
    #position dosyanın son yerindeki indisi tutar ve her turda dosyadaki yerle onu kıyaslayarak ilerleriz
    f = open("database.txt","r")
    f.seek(0,2)
    position=f.tell() 
    f.seek(0,0)
    N = 5
    j = 0
    #Matriste 1. sütün databasedeki resmin ismini 2. sütun 2 resim arasındaki mesafeyi tutar
    ans = [ [ " ",-1 ] for j in range(N) ]
    #Her seferinde new line a kadar okuma yaparız. database.txt dosyasına kayıt sırası isim,R,G,B,hue şeklindedir
    while(position!=f.tell()):
        ansRGB = 0
        name = f.readline() 
        R = []
        G = []
        B = []
        line = f.readline()
        #Okuduğumuz satırları integerlar halinde ayırarak arraye ekleriz
        for x in line.split():
            R.append(int(x))
        
        line = f.readline()
        for x in line.split():
            G.append(int(x))
        
        line = f.readline()
        for x in line.split():
            B.append(int(x))
        
        line = f.readline()

        #2 resim arasındaki RGB mesafeyi ölçeriz
        for i in range(256):
            ansRGB += math.sqrt( pow(histR[i]-R[i],2) + pow(histG[i]-G[i],2) + pow(histB[i]-B[i],2))
        #eğer mesafe matrise kayıtlı olan önceki mesafelerden küçük ise veya matrisin o gözünde -1 var ise  değeri matrise
        #ekleriz. Burada ilk indisten son indise gidildikçe mesafenin resimler arası mesafe değerleri artar.
        for i in range(N):
           if(ansRGB<ans[i][1] or ans[i][1]==-1 ):
              temp=ans[i][1]
              tempN=ans[i][0]
              ans[i][1]=ansRGB
              ans[i][0]=name
              for j in range(i+1,N):
                  temp2=ans[j][1]
                  tempN2=ans[j][0]
                  ans[j][0]=tempN
                  ans[j][1]=temp
                  tempN=tempN2
                  temp=temp2
              break
                  
    f.close()
    return ans

def findWhichHUE(str):
    #Burada resmi 3 boyutlu bir matris olarak data'ya alırız
    image = Image.open(str)
    data = asarray(image)
    
     #Satır ve sütun sayılarını row ve column'a alırız
    row = len(data)
    column = len(data[0])
    
     #360 birimlik Hue histogramına ve rgb2hue matrisine başlangıç olarak 0 atarız
    Hue = array([[0]*column]*row)
    histH = array([0]*360)
    
    #RGB değerlerinde Hue değerini elde ederiz
    for i in range(row):
        for j in range(column):
            r = data[i][j][0]/255.0
            g = data[i][j][1]/255.0
            b = data[i][j][2]/255.0
            
            cmax = max(r,g,b)
            cmin = min(r,g,b)
            diff = cmax-cmin
            if(cmax == cmin):
                Hue[i][j] = 0
            elif(cmax == r):
                Hue[i][j] = (60*(g-b)/diff+360)%360    
            elif(cmax == g):
                Hue[i][j] = (60*(b-r)/diff+120)%360
            elif(cmax == b):
                Hue[i][j] = (60*(r-g)/diff+240)%360
    
    #Elde ettiğimiz Hue değerlerinin histogramını alırız
    for i in range(row):
        for j in range(column):
            histH[Hue[i][j]] += 1
            
    #Resimlerin boyutları birbirlerinden farklı olacağı için bunları aynı hizaya çekeriz
    for i in range(360):
        histH[i] = 8192*histH[i]/(row*column) 
    
    #Database e erişmek için database.txt dosyasını reading modda açarız
    #position dosyanın son yerindeki indisi tutar ve her turda dosyadaki yerle onu kıyaslayarak ilerleriz
    f = open("database.txt","r")
    f.seek(0,2)
    position=f.tell()
    f.seek(0,0)
    N = 5
    j = 0
    #Matriste 1. sütün databasedeki resmin ismini 2. sütun 2 resim arasındaki mesafeyi tutar
    ans = [ [ " ",-1 ] for j in range(N) ]
    #Her seferinde new line a kadar okuma yaparız. database.txt dosyasına kayıt sırası isim,R,G,B,hue şeklindedir
    while(position!=f.tell()):
        ansHUE = 0
        name = f.readline() 
        H = []
        line = f.readline()
        
        line = f.readline()
        
        line = f.readline()
        
        line = f.readline()
        #Okuduğumuz satırları integerlar halinde ayırarak arraye ekleriz
        for x in line.split():
            H.append(int(x))
        
        #2 resim arasındaki Hue mesafeyi ölçeriz
        for i in range(360):
            ansHUE += math.sqrt( pow(histH[i]-H[i],2)) 
        
        #eğer mesafe matrise kayıtlı olan önceki mesafelerden küçük ise veya matrisin o gözünde -1 var ise  değeri matrise
        #ekleriz. Burada ilk indisten son indise gidildikçe mesafenin resimler arası mesafe değerleri artar.
        for i in range(N):
           if(ansHUE<ans[i][1] or ans[i][1]==-1 ):
              temp=ans[i][1]
              tempN=ans[i][0]
              ans[i][1]=ansHUE
              ans[i][0]=name
              for j in range(i+1,N):
                  temp2=ans[j][1]
                  tempN2=ans[j][0]
                  ans[j][0]=tempN
                  ans[j][1]=temp
                  tempN=tempN2
                  temp=temp2
              break                  
    f.close()
    return ans        

def histOfImage(str):
    #Burada resmi 3 boyutlu bir matris olarak data'ya alırız
    image = Image.open(str)
    data = asarray(image)
    
    #Satır ve sütun sayılarını row ve column'a alırız
    row = len(data)
    column = len(data[0])

    #360 birimlik Hue histogramına, rgb2hue matrisine ve RGB histogramlarına başlangıç olarak 0 atarız
    Hue = array([[0]*column]*row)
    
    histR = array([0]*256)
    histG = array([0]*256)
    histB = array([0]*256)
    histH = array([0]*360)
    
    #Hue transform
    for i in range(row):
        for j in range(column):
            r = data[i][j][0]/255.0
            g = data[i][j][1]/255.0
            b = data[i][j][2]/255.0
            
            cmax = max(r,g,b)
            cmin = min(r,g,b)
            diff = cmax-cmin
            if(cmax == cmin):
                Hue[i][j] = 0
            elif(cmax == r):
                Hue[i][j] = (60*(g-b)/diff+360)%360    
            elif(cmax == g):
                Hue[i][j] = (60*(b-r)/diff+120)%360
            elif(cmax == b):
                Hue[i][j] = (60*(r-g)/diff+240)%360
    
    #RGB değerlerinde Hue değerini elde ederiz 
    for i in range(row):
        for j in range(column):
            histR[data[i][j][0]] += 1
            histG[data[i][j][1]] += 1
            histB[data[i][j][2]] += 1
            histH[Hue[i][j]] += 1
            
     #Resimlerin boyutları birbirlerinden farklı olacağı için bunları aynı hizaya çekeriz
    for i in range(256):
        histR[i] = 8192*histR[i]/(row*column)
        histG[i] = 8192*histG[i]/(row*column)
        histB[i] = 8192*histB[i]/(row*column)
    
    for i in range(360):
        histH[i] = 8192*histH[i]/(row*column)    
        
    #Resimlerin isim, R,G,B ve Hue histgoramları sırayla ekleneceği için dosyayı append modda açarız
    f = open("database.txt","a")
    f.write(str+"\n")
    for line in histR:
        f.write("%d "%line)
    f.write('\n')
    for line in histG:
        f.write("%d "%line)
    f.write('\n')
    for line in histB:
        f.write("%d "%line)
    f.write('\n')
    for line in histH:
        f.write("%d "%line)
    f.write('\n')
    f.close()
    
#Program baştan çalıştırıldığında databaseyi sıfırlama için database.txt dosyasını sıfırlamak için
#dosyayı yazma modunda açıp kapatıyorum
p = open("database.txt","w")
p.close()

#Burada camel, dog, dolphin, giraffe, goose ve horse klasörlerindeki 
#resimlerin histogramlarını çıkarıp database'e işliyorum

files = ["camel/","dog/","dolphin/","giraffe/","goose/","horse/"]

for i in files:
    for img in glob.glob(i+"*.jpg"):
        histOfImage(img)
  
for img in glob.glob("test/*.jpg"):
    print(img)
    print("RGB")
    ans=findWhichRGB(img)
    print(ans)
    print("HUE")
    ans=findWhichHUE(img)
    print(ans)
    print("\n")