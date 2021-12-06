import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from operator import itemgetter
from datetime import datetime

def addRowsCols(matrice,width,height) :
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
    return matrice,width,height

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


def Descripteur(matrice,width,height):
    descipteur=[]    
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


def comparerFace(imagec,image0,folderpath):
    
    compareMse=[]

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image0, 1.1, 4)
    if(len(faces)==0):
        return []
    for (x, y, w, h) in faces:
        image1=image0[y:h+y,x:w+x]
        cv2.rectangle(imagec, (x, y), (x + w, y + h), (255, 0, 0), 2)  
    plt.figure("Face Detection")
    plt.imshow(np.array(imagec))
    plt.axis("off")    
    width=128
    height=128



    image1 = cv2.resize(image1, (width,height))
    matrice1 = np.asarray(image1)
    matrice1,width,height=addRowsCols(matrice1,width,height)        
    desc1,lbp1=Descripteur(matrice1,width,height)

    list_files=[]
    for path in os.listdir(folderpath):
        full_path = os.path.join(folderpath, path)
        if os.path.isfile(full_path):
            list_files.append(full_path)


    list_faces=[]
    width=128
    height=128
    for j in range(0,len(list_files)):
        image01=cv2.imread(list_files[j],0)
        faces2 = face_cascade.detectMultiScale(image01, 1.1, 4)
        for (x, y, w, h) in faces2:
            image=image01[y:h+y,x:w+x]
            image = cv2.resize(image, (width,height))
            list_faces.append(np.asarray(image)) 
    


    fig=plt.figure("LBP Test",figsize=(10,7))
    n=len(list_faces)+1
    fig.add_subplot(1,n,1)
    plt.imshow(cv2.hconcat([lbp1,image1]),cmap="gray")
    plt.axis("off")

    for i in range(n-1):
        matrice2 = list_faces[i]
        width=128
        height=128
        matrice2,width,height=addRowsCols(matrice2,width,height)           
        desc2,lbp2=Descripteur(matrice2,width,height)
        fig.add_subplot(1,n,i+2)
        plt.imshow(cv2.hconcat([lbp2,list_faces[i]]),cmap="gray")
        mse=round(Sim(desc1,desc2),2)
        if(i>=len(list_files)):
            list_files.append(list_files[i-1])
        k={'name':list_files[i]}
        k['MSE']=mse
        compareMse.append(k)
        plt.title("MSE : " + str(mse))

        plt.axis("off")
    
    compareMse = sorted(compareMse, key=itemgetter('MSE'))
    fig.tight_layout()
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return compareMse

                  
def showcam():
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while(True):
        (_, image) = webcam.read()
        image0 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(image0, 1.1, 4)
        for (x, y, w, h) in faces:
            image1=image0[y:h+y,x:w+x]
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  
        cv2.imshow('frame',image) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    webcam.release()
    cv2.destroyAllWindows()    

def showcam2():
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    count = 0
    while count < 3:
        (_, image) = webcam.read()
        cv2.imwrite("test2/image"+datetime.now().strftime("%d %m %Y %H %M %S")+".png",np.asarray(image))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print( 
          count)
        list_mse=comparerFace(image,gray,'test2/')  
        for i in range(len(list_mse)):   
            print(list_mse[i])
        count += 1
    webcam.release()
    cv2.destroyAllWindows()

showcam()