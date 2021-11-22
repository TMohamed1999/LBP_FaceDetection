import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image







                     
def addRowsCols(matrice) :
    global width,height
    col=[[row[0]] for row in matrice]
    matrice = np.append(col, matrice, axis=1)
    width+=1
    col=[[row[width-1]] for row in matrice]
    matrice = np.append(matrice,col, axis=1)
    width+=1

    ligne=[val for val in matrice[0]]
    matrice = np.append([ligne],matrice, axis=0)
    height+=1
    ligne=[val for val in matrice[height-1]]
    matrice = np.append(matrice,[ligne], axis=0)
    height+=1
    return matrice

def LBP_8X8(matrice,lbp,posi,posj):
     hist_y=np.zeros(256)
     for i in range(posi,posi+8):
        for j in range(posj,posj+8):
            xj=j-1
            yi=i-1
            centre=matrice[yi+1][xj+1]
            lbp_val=int(matrice[yi][xj]>=centre)+ int(matrice[yi][xj+1]>=centre)*2 + int(matrice[yi][xj+2]>=centre)*4 + int(matrice[yi+1][xj+2]>=centre)*8 + int(matrice[yi+2][xj+2]>=centre)*16+ int(matrice[yi+2][xj+1]>=centre)*32 + int(matrice[yi+2][xj]>=centre)*64 + int(matrice[yi+1][xj]>=centre)*128
            lbp[yi][xj]=lbp_val
            hist_y[lbp_val]+=1
            

     return hist_y;    


def Descripteur(matrice):
    descipteur=[]    
    global width,height
    lbp=np.zeros([height-2,width-2],dtype=np.uint8)
    for i in range(1,height-1,8):
        for j in range(1,width-1,8) :
            hist_y=LBP_8X8(matrice,lbp,i,j)
            [descipteur.append(v) for v in hist_y]
    return descipteur,lbp        


def Sim(desc1,desc2):
    mse=0;
    for i in range(np.size(desc1)):
        mse=mse+(desc1[i]-desc2[i])**2 
    return mse/len(desc1); 





face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

folder='test2/'
list_files=[]
for path in os.listdir(folder):
    full_path = os.path.join(folder, path)
    if os.path.isfile(full_path):
          list_files.append(full_path)


ch=0
image0=cv2.imread(list_files[ch],0)
faces = face_cascade.detectMultiScale(image0, 1.1, 4)
for (x, y, w, h) in faces:
    image1=image0[y:h+y,x:w+x]


width=128
height=128



image1 = cv2.resize(image1, (width,height))
matrice1 = np.asarray(image1)
matrice1=addRowsCols(matrice1)        
desc1,lbp1=Descripteur(matrice1)



list_images=[]
width=128
height=128
for j in range(0,len(list_files)):
    image01=cv2.imread(list_files[j],0)
    faces2 = face_cascade.detectMultiScale(image01, 1.1, 4)
    for (x, y, w, h) in faces2:
        image=image01[y:h+y,x:w+x]
        image = cv2.resize(image, (width,height))
        list_images.append(np.asarray(image)) 



fig=plt.figure("LBP Test",figsize=(100,100))
n=len(list_images)+1
fig.add_subplot(n,1,1)
plt.imshow(cv2.hconcat([lbp1,image1]),cmap="gray")



for i in range(n-1):
    matrice2 = list_images[i]
    width=128
    height=128
    matrice2=addRowsCols(matrice2)           
    desc2,lbp2=Descripteur(matrice2)
    fig.add_subplot(n,1,i+2)
    plt.imshow(cv2.hconcat([lbp2,list_images[i]]),cmap="gray")
    print("Compare  "+list_files[i]+ " TO " + list_files[i]+"    :"+str(Sim(desc1,desc2)))


fig.tight_layout()
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

