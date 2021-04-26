from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import random
import re
import time
import os
import glob, os
import copy
import math


from utils import *

UNKNOWN = -1
path='D:/psc/YOLO/Number_detect/annot41.txt'
path3='D:/psc/YOLO/Number_detect/'
name2="DJI_0273" #139
aLim=1
bLim=505 #505
fps=30 #24

def dumbCalcul():
    f=open(path)
    lignes=f.readlines()
    categories=[]
    dicOccur={}
    for ligne in lignes:
        if ligne=="":
            continue
        cat=ligne.split(' ')[4]
        if cat not in categories:
            categories.append(cat)
            dicOccur[cat]=1
        dicOccur[cat]+=1
    for cat in categories:
        print(cat+" : "+str(dicOccur[cat]))
    f.close()

def distVar(p1,p2):
    n=len(p1)
    s=0
    for i in range(4):
        s+=(p1[i]-p2[i])**2
    s+=(((p1[2]-p1[0])-(p2[2]-p2[0]))**2+((p1[3]-p1[1])-(p2[3]-p2[1]))**2)/alpha
    return s**0.5

def dumbCalculWithDist(seuil=1,upTo=5):
    f=open(path)
    lignes=f.readlines()
    categories=[]
    dicOccur={}
    #[cat,number of bbox,point 4-coords, frame]
    annot={}
    for ligne in lignes:
        spl=ligne.split(' ')
        cat=spl[4]
        frame = int(spl[-1].split('_')[-1])
        if frame not in annot:
            annot[frame]=[]
        annot[frame].append([cat,UNKNOWN,float(spl[0]),float(spl[1]),float(spl[2]),float(spl[3])])
        if cat not in categories:
            categories.append(cat)
            dicOccur[cat]=1
        dicOccur[cat]+=1
    
    f.close()
    keys=annot.keys()
    keys=[k for k in keys]
    keys.sort()
    current_count={}
    last_ones={}
    for cat in categories:
        current_count[cat]=0
    for i in range(len(keys)):
        for j in range(len(annot[keys[i]])):
            if current_count[annot[keys[i]][j][0]]==0:
                current_count[annot[keys[i]][j][0]]+=1
                annot[keys[i]][j][1]=1
                last_ones[(annot[keys[i]][j][0],1)]=[[i,j]]
                continue
            T=[]
            for k in range(current_count[annot[keys[i]][j][0]]):
                M=[last_ones[(annot[keys[i]][j][0],current_count[annot[keys[i]][j][0]]-k)][h] for h in range(len(last_ones[(annot[keys[i]][j][0],current_count[annot[keys[i]][j][0]]-k)])) if last_ones[(annot[keys[i]][j][0],current_count[annot[keys[i]][j][0]]-k)][h][0]>=i-upTo]
                T+=M
                if M==[]:
                    break
            H=[distVar(annot[keys[i]][j][2:],annot[keys[T[h][0]]][T[h][1]][2:]) for h in range(len(T)) if distVar(annot[keys[i]][j][2:],annot[keys[T[h][0]]][T[h][1]][2:])<=seuil]
            print(H)
            ind=argmin(H)
            if ind==-1:
                current_count[annot[keys[i]][j][0]]+=1
                annot[keys[i]][j][1]=current_count[annot[keys[i]][j][0]]
                last_ones[(annot[keys[i]][j][0],current_count[annot[keys[i]][j][0]])]=[[i,j]]
            else:
                annot[keys[i]][j][1]=annot[keys[T[ind][0]]][T[ind][1]][1]
                last_ones[(annot[keys[i]][j][0],current_count[annot[keys[i]][j][0]])].append([i,j])
                
            if(annot[keys[i]][j][1] == -1):
                print("prob")
            
    convVid(annot, categories,'D:/psc/YOLO/Number_detect/Images_to_vid/',last_ones,keys)

def ridOfToAvoid(annot, categories, last_ones ,keys, path2):
    Index=[]
    numRidAlready=0
    for m in last_ones:
        value=keys[last_ones[m][0][0]]
        if value>=aLim and value<=bLim:
            Index.append([value,last_ones[m][0][1]])
        
        value=keys[last_ones[m][-1][0]]
        if value>=aLim and value<=bLim:
            Index.append([value,last_ones[m][-1][1]])
        
        longeur = len(last_ones[m])
        value=keys[last_ones[m][longeur//2][0]]
        if value>=aLim and value<=bLim:
            Index.append([value,last_ones[m][longeur//2][1]])
    
    g=glob.glob(path3+"ToAvoid/*.png")
    names=[path3+'ToAvoid/'+k.split('\\')[-1] for k in g]
    
    g=glob.glob(path3+"ToLeave/*.png")
    namesLeave=[path3+'ToLeave/'+k.split('\\')[-1] for k in g]
    skip=False
    for l in Index:
        if annot[l[0]][l[1]][0]=='RidOf':
            continue
        img1 = cv2.imread(path2+name2+"_fr_"+str(l[0])+".jpg")
        img1=img1[round(annot[l[0]][l[1]][3]):round(annot[l[0]][l[1]][5]),round(annot[l[0]][l[1]][2]):round(annot[l[0]][l[1]][4])]
        categ=annot[l[0]][l[1]][0]
        num=annot[l[0]][l[1]][1]
        
        for name in namesLeave:
            img2 = cv2.imread(name)
            dis=distImagesConvol2(img1,img2)
            
            print([categ,num,'leave'])
            print(dis)
            if(dis<=seuilLeave):
                print('left')
                
                skip=True
            print()
        
        for name in names:
            if skip:
                skip=False
                break
            img2 = cv2.imread(name)
            dis=distImagesConvol2(img1,img2)
            
            print([categ,num])
            print(dis)
            if(dis<=seuilToRid):
                print('yep')
                print()
                
                
                numRidAlready+=1
                if 'RidOf' not in categories:
                    categories.append('RidOf')
                
                
                
                for t in last_ones[(categ,num)]:
                    
                    annot[keys[t[0]]][t[1]][1]=numRidAlready
                    annot[keys[t[0]]][t[1]][0]='RidOf'
                numMaxCat=num
                while (categ, numMaxCat+1) in last_ones:
                    numMaxCat+=1
                #print(annot[keys[tab2[0][0]]][tab1[0][1]])
            #print(num2)
                if numMaxCat>=num+1:
                    for j in range(num+1,numMaxCat+1):
                        for k in last_ones[(categ,j)]:
                            annot[keys[k[0]]][k[1]][1]=j-1
                        last_ones[(categ,j-1)]=last_ones[(categ,j)]
                    
                del last_ones[(categ,numMaxCat)]
                break
            print()
    return (annot,categories,last_ones)
        
        #seuilToRid
        
    
        #if last_ones[0]
    
      
def distImagesConvol2(img1,img2):
    height1, width1, layers1 = img1.shape
    height2, width2, layers2 = img2.shape
    
    stepX1, stepX2, stepY1, stepY2=max(1,width1/width2),max(1,width2/width1),max(1,height1/height2),max(1,height2/height1)
    s=0 #sum ai*bi
    smax1=0 #sum ai^2
    smax2=0 #sum bi^2
    moy1=0
    moy2=0
    n=min(height1,height2)*width2
    # "smallest image", width wise, is 2
    if stepX2!=1:
        return distImagesConvol(img2,img1)
    for cursX in range(width2):
        for cursY in range(min(height1,height2)):
            #tab1=img1[round(stepY1*cursY):round(stepY1*(cursY+1))][round(stepX1*cursX):round(stepX1*(cursX+1))]
            tab1=np.array([np.mean(img1[round(stepY1*cursY):round(stepY1*(cursY+1)), round(stepX1*cursX):round(stepX1*(cursX+1)), k]) for k in range(layers1)])
            tab2=np.array([np.mean(img2[round(stepY2*cursY):round(stepY2*(cursY+1)), round(stepX2*cursX):round(stepX2*(cursX+1)), k]) for k in range(layers1)])
            s+=np.sum(tab1*tab2)/100000000
            smax1+=np.sum(tab1*tab1)/100000000
            smax2+=np.sum(tab2*tab2)/100000000
            moy1+=np.sum(tab1)/100000000
            moy2+=np.sum(tab2)/100000000
    moy1=moy1/n
    moy2=moy2/n
    smax1-=(moy1**2)*n
    smax2-=(moy2**2)*n
    s-=n*moy1*moy2
    return 1-s/((smax1*smax2)**0.5)
    
def drawPredicFPSVitBefore(annot, categories, last_ones ,keys, upToLimit=100):
    times={}
    for m in last_ones:
        times[m]=[keys[last_ones[m][0][0]],keys[last_ones[m][-1][0]]]
    keys2=[k for k in last_ones]
    LookUpBfr=20
    for m in last_ones:
        (a,b)=m
        if('predictBFRVIt '+a not in categories):
            categories.append('predictBFRVIt '+a)
        i0=keys[last_ones[m][-1][0]]
        vect=annot[i0][last_ones[m][-1][1]][:]
        vect[0]='predictBFRVIt '+a
        for k in range(1,upToLimit):
            if i0+k not in annot:
                annot[i0+k]=[]
            (vx2,vy2)=calculVitMoyBetweenFr(annot, categories, last_ones, keys, keys2, times, max(0,i0+k-LookUpBfr), i0+k)
            if vx2!=None:
                (vx,vy)=(vx2,vy2)
            vect[2]+=vx
            vect[4]+=vx
            vect[3]+=vy
            vect[5]+=vy
            vect[6]=i0+k
            annot[i0+k].append(vect[:])
    return (annot,categories,last_ones)
        
    
def distImagesConvol(img1,img2):
    height1, width1, layers1 = img1.shape
    height2, width2, layers2 = img2.shape
    
    stepX1, stepX2, stepY1, stepY2=max(1,width1/width2),max(1,width2/width1),max(1,height1/height2),max(1,height2/height1)
    s=0 #sum ai*bi
    smax1=0 #sum ai^2
    smax2=0 #sum bi^2
    # "smallest image", width wise, is 2
    if stepX2!=1:
        return distImagesConvol(img2,img1)
    for cursX in range(width2):
        for cursY in range(min(height1,height2)):
            #tab1=img1[round(stepY1*cursY):round(stepY1*(cursY+1))][round(stepX1*cursX):round(stepX1*(cursX+1))]
            tab1=np.array([np.mean(img1[round(stepY1*cursY):round(stepY1*(cursY+1)), round(stepX1*cursX):round(stepX1*(cursX+1)), k]) for k in range(layers1)])
            tab2=np.array([np.mean(img2[round(stepY2*cursY):round(stepY2*(cursY+1)), round(stepX2*cursX):round(stepX2*(cursX+1)), k]) for k in range(layers1)])
            s+=np.sum(tab1*tab2)/100000000
            smax1+=np.sum(tab1*tab1)/100000000
            smax2+=np.sum(tab2*tab2)/100000000
    
    return 1-abs(s/((smax1*smax2)**0.5))
        
            
def convVid(annot, categories, path2, last_ones,keys):
    times={}
    for m in last_ones:
        (a,b)=m
        temp=last_ones[m][:]
        temp.sort()
        if a not in times:
            times[a]=[]
        times[a].append([keys[temp[0][0]],keys[temp[-1][0]]])
    count4={}
    for m in categories:
        count4[m]=0
    
    for a in times:
        count4[a]=len([x for x in times[a] if x[1]<aLim])
    names_arr = [path2+name2+"_fr_"+str(k)+'.jpg' for k in range(aLim,bLim)]
    size,width,height=(0,0),0,0
    colors={}
    colors[categories[0]]=(255,0,0)
    if len(categories)>=2:
        colors[categories[1]]=(0,0,255)
    if len(categories)>=3:
        colors[categories[2]]=(0,255,0)
    if len(categories)>=4:
        for i in range(len(categories)-3):
            colors[categories[i+3]]=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
    colors["discarded"]=(0,255,255)
    colors['artVitMvmt']=(203,192,255)
    colors['artPred']=(128,0,128)
    colors['RidOf']=(0,255,0)
    img = cv2.imread(names_arr[0])
    #print(img)
    height, width, layers = img.shape
    size = (width,height)
    img=[]
    out = cv2.VideoWriter('D:/psc/YOLO/Number_detect/result/result_'+name2+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    #annot={}
    for filename in names_arr:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        frame=int(filename.split('.')[-2].split('_')[-1])
        nb=1
        print(frame)
        if frame in annot and annot[frame]!=[]:
            for tab in annot[frame]:
                #if(tab[0][:4] != 'Fill'):
               #     continue
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                thickness = 2
                #print(frame)
                #print(tab)
                #print()
                name=tab[0]
                if name=="FillIn_json":
                    name="Plastic Bottle"
                img = cv2.rectangle(img, (int(tab[2]),int(tab[3])), (int(tab[4]),int(tab[5])), colors[tab[0]], 10)
                l=cv2.getTextSize(name+' '+str(tab[1]-count4[tab[0]]), font, fontScale, thickness)[0][0]
                
                cv2.rectangle(img, (int(tab[2])-5, max(0,int(tab[3])-35)), (int(tab[2])+l+5, max(0,int(tab[3])-35)+35), colors[tab[0]], -1)
                yDep=10
                img = cv2.putText(img, name+' '+str(tab[1]-count4[tab[0]]), (int(tab[2]),max(0,int(tab[3])-35)+25), font, fontScale, (0,0,0), thickness, cv2.LINE_AA)
        out.write(img)
                #if nb<=5:
                    #window_name = 'image'
  
# Using cv2.imshow() method  
# Displaying the image  
                   # cv2.imshow(window_name, img) 
  
#waits for user to press any key  
#(this is necessary to avoid Python kernel form crashing) 
                  #  cv2.waitKey(0)  
  
#closing all open windows  
                  #  cv2.destroyAllWindows()
                  #  nb+=1 
 
        #a=1
    cv2.destroyAllWindows()
    out.release()
    
            
def argmin(T):
    if T==[] or len(T)<=0:
        return -1
    ind=0
    min=T[0]
    for j in range(len(T)):
        if T[j]<min:
            min=T[j]
            ind=j
    return ind
    
def mini(T):
    if T==[]:
        return -1
    ind=0
    min=T[0]
    for j in range(len(T)):
        if T[j]<min:
            min=T[j]
            ind=j
    return min
    
#no 2 same objects in the same fps, removes objects with few occurences, vit either False or table
def dumbCalculWithDist2(seuil=1,upTo=5,removeUpTo=0,vit=False):
    Vx=0
    Vy=0
    if type(vit)!=bool:
        Vx=vit[0]
        Vy=vit[1]
    f=open(path)
    lignes=f.readlines()
    categories=[]
    dicOccur={}
    #[cat,number of bbox,point 4-coords,your i]
    annot={}
    for ligne in lignes:
        spl=ligne.split(' ')
        cat=spl[4]
        frame = int(spl[-1].split('_')[-1])
        if frame not in annot:
            annot[frame]=[]
        annot[frame].append([cat,UNKNOWN,float(spl[0]),float(spl[1]),float(spl[2]),float(spl[3]),frame])
        if cat not in categories:
            categories.append(cat)
            dicOccur[cat]=1
        dicOccur[cat]+=1
    
    f.close()
    keys=annot.keys()
    keys=[k for k in keys]
    keys.sort()
    current_count={}
    last_ones={}
    for cat in categories:
        current_count[cat]=0
    for i in range(len(keys)):
        possi={}
        taken=[]
        for j in range(len(annot[keys[i]])):
            if current_count[annot[keys[i]][j][0]]==0:
                current_count[annot[keys[i]][j][0]]+=1
                annot[keys[i]][j][1]=1
                last_ones[(annot[keys[i]][j][0],1)]=[[i,j]]
                continue
            T=[]
            for k in range(current_count[annot[keys[i]][j][0]]):
                M=[last_ones[(annot[keys[i]][j][0],current_count[annot[keys[i]][j][0]]-k)][h] for h in range(len(last_ones[(annot[keys[i]][j][0],current_count[annot[keys[i]][j][0]]-k)])) if (last_ones[(annot[keys[i]][j][0],current_count[annot[keys[i]][j][0]]-k)][h][0]>=i-upTo and last_ones[(annot[keys[i]][j][0],current_count[annot[keys[i]][j][0]]-k)][h][0]<=i-1 and [annot[keys[i]][j][0],current_count[annot[keys[i]][j][0]]-k] not in taken)]
                T+=M
            #T=set(T)
            #T=[k for k in T]
            
            #H=[distVar(annot[keys[i]][j][2:],annot[keys[T[h][0]]][T[h][1]][2:]) for h in range(len(T)) if distVar(annot[keys[i]][j][2:],annot[keys[T[h][0]]][T[h][1]][2:])<=seuil]
            H2=[annot[keys[T[h][0]]][T[h][1]] for h in range(len(T)) if distVar(decale(annot[keys[i]][j][2:],[-1*Vx*keys[i],-1*Vy*keys[i]]),decale(annot[keys[T[h][0]]][T[h][1]][2:],[-1*Vx*keys[T[h][0]],-1*Vy*keys[T[h][0]]]))<=seuil]
            #H=[distVar(decale(annot[keys[i]][j][2:],[-1*Vx*i,-1*Vy*i]),decale(annot[keys[T[h][0]]][T[h][1]][2:],[-1*Vx*T[h][0],-1*Vy*T[h][0]])) for h in range(len(T)) if distVar(decale(annot[keys[i]][j][2:],[-1*Vx*i,-1*Vy*i]),decale(annot[keys[T[h][0]]][T[h][1]][2:],[-1*Vx*T[h][0],-1*Vy*T[h][0]]))<=seuil]
            H=[distVar(decale(annot[keys[i]][j][2:],[-1*Vx*keys[i],-1*Vy*keys[i]]),decale(h[2:],[-1*h[6]*Vx,-1*h[6]*Vy])) for h in H2]
            #print(H)
            ind=argmin(H)
            if ind==-1:
                possi[j]=[H2,ind,0,[]]
            else:
                
                possi[j]=[H2,ind,H[ind],[ annot[keys[T[ind][0]]][T[ind][1]][0] ,annot[keys[T[ind][0]]][T[ind][1]][1] ] ]
            if ind==-1:
                current_count[annot[keys[i]][j][0]]+=1
                annot[keys[i]][j][1]=current_count[annot[keys[i]][j][0]]
                last_ones[(annot[keys[i]][j][0],current_count[annot[keys[i]][j][0]])]=[[i,j]]
                taken.append([annot[keys[i]][j][0],current_count[annot[keys[i]][j][0]]])
            #else:
             #   annot[keys[i]][j][1]=annot[keys[T[ind][0]]][T[ind][1]][1]
              #  last_ones[(annot[keys[i]][j][0],current_count[annot[keys[i]][j][0]])].append([i,j])
        
        Indices=[k for k in possi if possi[k][0]!=[]] #good j
        m=len(Indices)
        alreadyPresent=[]
        co=0
        while(m>=1):
            iMin=argmin([possi[h][2] for h in Indices])
            annot[keys[i]][Indices[iMin]][1]=possi[Indices[iMin]][0][ possi[Indices[iMin]][1] ][1]
            last_ones[(possi[Indices[iMin]][0][ possi[Indices[iMin]][1] ][0],possi[Indices[iMin]][0][ possi[Indices[iMin]][1] ][1])].append([i,Indices[iMin]])
            co+=1
            
            taken.append([possi[Indices[iMin]][0][ possi[Indices[iMin]][1] ][0],possi[Indices[iMin]][0][ possi[Indices[iMin]][1] ][1]])
            loc=[possi[Indices[iMin]][0][ possi[Indices[iMin]][1] ][0],possi[Indices[iMin]][0][ possi[Indices[iMin]][1] ][1]]
          #  if i==134:
           #     print(possi[Indices[iMin]][0][ possi[Indices[iMin]][1] ][1])
            possi[Indices[iMin]]=[[],-1,0,[]]
            #Indices=[ind in Indices if ind!=iMin]
            #m=len(Indices)
            
            
      #          print(Indices)
       #         print(annot[keys[i]][Indices[iMin]])
            
            for k in range(m):
                if k==iMin:
                    continue
                j=Indices[k]
                #H2=[[h2 for h2 in h[0] if [h2[0],h2[1]] != possi[Indices[iMin]][3]] for h in possi[j]]
                
                H2=[possi[j][0][indx] for indx in range(len(possi[j][0])) if [ possi[j][0][indx][0], possi[j][0][indx][1] ] !=loc ]
              #  if i==134:
               #     print()
                    
                #    print(H2)
                H2=[h2 for h2 in H2 if h2!=[]]
              #  if i==134:
               #     print(loc[3])
                #    print(H2)
                

                H=[distVar(decale(annot[keys[i]][j][2:],[-1*annot[keys[i]][j][6]*Vx,-1*annot[keys[i]][j][6]*Vy]),decale(h[2:],[-1*Vx*h[6],-1*Vy*h[6]])) for h in H2]
                ind=argmin(H)
                if ind==-1:
                    current_count[annot[keys[i]][j][0]]+=1
                    annot[keys[i]][j][1]=current_count[annot[keys[i]][j][0]]
                    last_ones[(annot[keys[i]][j][0],current_count[annot[keys[i]][j][0]])]=[[i,j]]
                    possi[j]=[H2,ind,0,[]]
                else:
                    possi[j]=[H2,ind,H[ind],[ H2[ind][0],H2[ind][1] ] ]
            Indices=[k for k in possi if possi[k][0]!=[]]
            m=len(Indices)
                    
                
    takenIndex=[]
    real_last_ones={}
    if removeUpTo!=0:
        #last_ones does not separate useless
        
        k=1
        bo=False
        for m in last_ones:
            
            if len(last_ones[m])<=removeUpTo:
                takenIndex.append(m)
                bo=True
                for l in range(len(last_ones[m])):
                    #print(last_ones[m])
                    annot[keys[last_ones[m][l][0]]][last_ones[m][l][1]][0]="discarded"
                    annot[keys[last_ones[m][l][0]]][last_ones[m][l][1]][1]=k
                k+=1
        if bo:
            categories.append('discarded')
        for m in last_ones:
            #quadratique bcs little data
            if m not in takenIndex :
                (a,b)=m
                for l in range(len(last_ones[m])):
                    k=annot[keys[last_ones[m][l][0]]][last_ones[m][l][1]][1]
                    annot[keys[last_ones[m][l][0]]][last_ones[m][l][1]][1]=len([(a,h) for h in range(1,k+1) if (a,h) not in takenIndex])
                    #print([a,b])
                    #print(len([(a,h) for h in range(1,k+1) if (a,h) not in takenIndex]))
                    #print(m)
                    #print([k,annot[keys[last_ones[m][l][0]]][last_ones[m][l][1]][1]])
                #print(m)
                #print(last_ones [m])
    
        for m in last_ones:
            if m in takenIndex:
                continue
            (a,b)=m
            i0=last_ones[m][0][0]
            j0=last_ones[m][0][1]
            real_last_ones[(a,annot[keys[i0]][j0][1])]=last_ones[m]
    real_count={}
    last_ones=real_last_ones
    
    for k in current_count:
        real_count[k]=len([(a,h) for h in range(1,current_count[k]+1) if (a,h) not in takenIndex])
    if type(vit)==bool or vit[2]=='begin':
        
        copyAnnot=annot
        copyLastOnes=last_ones
        #(copyAnnot,categories) = FillInHoles(copyAnnot, categories, copyLastOnes ,keys)
        
        (Vx,Vy) = calculVitMoyBetweenFr(annot, categories, last_ones, keys, [], [], aLim, bLim, giveZero=True)
        
        #(Vx,Vy) = calculVit2(copyAnnot, copyLastOnes, keys)
        (annot,categories,last_ones) = dumbCalculWithDist2(seuil,upTo,removeUpTo,[Vx,Vy, False])
        print([Vx,Vy])
        
        copyAnnot=annot
        copyLastOnes=last_ones
        #(copyAnnot,categories) = FillInHoles(copyAnnot, categories, copyLastOnes ,keys)
        #(Vx,Vy) = calculVit2(copyAnnot, copyLastOnes, keys)
        (Vx,Vy) = calculVitMoyBetweenFr(annot, categories, last_ones, keys, [], [], aLim, bLim, giveZero=True)
        (annot,categories,last_ones) = dumbCalculWithDist2(seuil,upTo,removeUpTo,[Vx,Vy, False])
        print([Vx,Vy])
        
        #copyAnnot=copy.deepcopy(annot)
        #copyLastOnes=copy.deepcopy(last_ones)
        #(copyAnnot,categories) = FillInHoles(copyAnnot, categories, copyLastOnes ,keys)
        #(Vx,Vy) = calculVit2(copyAnnot, copyLastOnes, keys)
        #(annot,categories,last_ones) = dumbCalculWithDist2(seuil,upTo,removeUpTo,[Vx,Vy, False])
        
        #print([Vx,Vy])
        #print(current_count)
        #print(real_count)
       # print(len(takenIndex))
      #  print(last_ones)
        #print(takenIndex)
        #(annot,categories) = createMvmtWithV(annot, categories, last_ones ,keys, Vx, Vy)
        #(annot,categories) = createMvmtWithPred(annot, categories, last_ones ,keys)
        
        #(annot,categories) = createMvmtWithVMoyCatNum(annot, categories, last_ones ,keys)
        #(annot,categories) = createMvmtWithVMoyCatNum2(annot, categories, last_ones ,keys)
        (annot,categories,last_ones) = updateSameBBoxAppearLater(annot, categories, last_ones ,keys)
        #(annot,categories,last_ones) = drawPredicFPSVitBefore(annot, categories, last_ones ,keys)
        
        #(annot,categories,last_ones) = ridOfToAvoid(annot, categories, last_ones ,keys,'D:/psc/YOLO/Number_detect/Images_to_vid/')
        (annot,categories) = FillInHoles(annot, categories, last_ones ,keys)
        #CreateForVeng(annot, categories, last_ones ,keys)
        convVid(annot, categories,'D:/psc/YOLO/Number_detect/Images_to_vid/',last_ones,keys)
    else:
        a=1
        if vit[2]:
            a=1
            (annot,categories,last_ones) = updateSameBBoxAppearLater(annot, categories, last_ones ,keys)

            (annot,categories) = FillInHoles(annot, categories, last_ones ,keys)
            convVid(annot, categories,'D:/psc/YOLO/Number_detect/Images_to_vid/',last_ones,keys)
    return (annot, categories,last_ones)
    
def updateSameBBoxAppearLater(annot, categories, last_ones ,keys):
    weGucci="""for m in last_ones:
        n=len(last_ones[m])
        for i in range(n):
            for j in range(i):
                if last_ones[m][i][0]==last_ones[m][j][0]:
                    print("prob")"""
    
    #print(any([]))
    times={}
    redo=False
    for m in last_ones:
        temp=last_ones[m][:]
        temp.sort()
        times[m]=[keys[temp[0][0]],keys[temp[-1][0]]]
    
    keys2=[k for k in last_ones]
    eventualLink=[[[a,b] for a in range(len(keys2))] for b in range(len(keys2))]
    eventualLink2=[]
    for m in eventualLink:
        eventualLink2+=m
    
    eventualLink=eventualLink2
    eventualLink=[a for a in eventualLink if (firstArgTuple(keys2[a[0]])==firstArgTuple(keys2[a[1]]) and times[keys2[a[0]]][1] < times[keys2[a[1]]][0] and times[keys2[a[1]]][0]-times[keys2[a[0]]][1]<=limMaxReappear)]
    
    
    
    dictDiff=[]    
    for kh in eventualLink:
        tab1=last_ones[keys2[kh[0]]][:]
        tab1.sort()
        a=keys[tab1[-3][0]]

        
        tab2=last_ones[keys2[kh[1]]][:]
        tab2.sort()
        b=keys[tab2[2][0]]
        (vx,vy)=calculVitMoyBetweenFr(annot, categories, last_ones, keys, keys2, times, a, b)
        
        i1=keys[tab1[-1][0]]        
        i2=keys[tab2[0][0]]
        
        j1=tab1[-1][1]        
        j2=tab2[0][1]
        
        (x1,y1)=calcCenter(annot[i1][j1][2:6])        
        (x2,y2)=calcCenter(annot[i2][j2][2:6])
        
        (vxreal,vyreal)=((x1-x2)/(i1-i2),(y1-y2)/(i1-i2))
        num=annot[keys[tab1[0][0]]][tab1[0][1]][1]
        num2=annot[keys[tab2[0][0]]][tab2[0][1]][1]
        #print()
        #print(annot[keys[tab1[0][0]]][tab1[0][1]])
        #print(annot[keys[tab2[0][0]]][tab1[0][1]])
        #print()
        #print(num2)
        if distVit(vxreal,vyreal,vx,vy,i1-i2)<=alpha_vit:
            dictDiff.append([distVit(vxreal,vyreal,vx,vy,i1-i2),kh[0],kh[1]])
            redo=True
    
    if redo:
        print("yep")
        dictDiff.sort()
        #print(keys2[dictDiff[0][1]],keys2[dictDiff[0][2]])
        #print(last_ones[keys2[dictDiff[0][1]]])
        #print(last_ones[keys2[dictDiff[0][2]]])
        #print()
        
        kh=dictDiff[0][1:3]
        tab1=last_ones[keys2[kh[0]]][:]
        tab1.sort()
        a=keys[tab1[-3][0]]

        
        tab2=last_ones[keys2[kh[1]]][:]
        tab2.sort()
        b=keys[tab2[2][0]]
        (vx,vy)=calculVitMoyBetweenFr(annot, categories, last_ones, keys, keys2, times, a, b)
        
        i1=keys[tab1[-1][0]]        
        i2=keys[tab2[0][0]]
        
        j1=tab1[-1][1]        
        j2=tab2[0][1]
        
        (x1,y1)=calcCenter(annot[i1][j1][2:6])        
        (x2,y2)=calcCenter(annot[i2][j2][2:6])
        
        (vxreal,vyreal)=((x1-x2)/(i1-i2),(y1-y2)/(i1-i2))
        num=annot[keys[tab1[0][0]]][tab1[0][1]][1]
        num2=annot[keys[tab2[0][0]]][tab2[0][1]][1]
        #print()
        #print(annot[keys[tab1[0][0]]][tab1[0][1]])
        #print(annot[keys[tab2[0][0]]][tab1[0][1]])
        #print()
        #print(num2)
        if distVit(vxreal,vyreal,vx,vy,i1-i2)<=alpha_vit:
            dictDiff+=[distVit(vxreal,vyreal,vx,vy,i1-i2),kh[0],kh[1]]
            redo=True
            
            for t in tab2:
                annot[keys[t[0]]][t[1]][1]=num
            print(num)
            print()
            #print(annot[keys[tab2[0][0]]][tab1[0][1]])
            last_ones[keys2[kh[0]]]+=tab2
            categ=firstArgTuple(keys2[kh[0]])
            numCateg=num2
            while (categ,numCateg+1) in last_ones:
                numCateg+=1
            #print(num2)
            if numCateg>=num2+1:
                for j in range(num2+1,numCateg+1):
                    for k in last_ones[(categ,j)]:
                        annot[keys[k[0]]][k[1]][1]=j-1
                    last_ones[(categ,j-1)]=last_ones[(categ,j)]
                    
            del last_ones[(categ,numCateg)]
            
    #print(eventualLink)
        
    #print(times)
    #print(eventualLink)
    if redo:
        #print(annot)
        weGucci="""Stop=False
        for m in last_ones:
            n=len(last_ones[m])
            (a,b)=m
            for i in range(n):
                if not Stop and annot[keys[last_ones[m][i][0]]][last_ones[m][i][1]][1]!=b:
                    print("prob")
                    print(annot[keys[last_ones[m][i][0]]][last_ones[m][i][1]][1],b)
                    print()
                    Stop=True"""
        updateSameBBoxAppearLater(annot, categories, last_ones ,keys)
    return (annot,categories,last_ones)
    
    #calculVitMoyCatNum(annot, categories, last_ones ,keys, m)

def distVit(vxreal,vyreal,vx,vy,diffFrames=0,AllVit=[]):
    #AllVit=[[vit1,frame1_1,frame1_2], [vit2,frame2_1,frame2_2], [vit3,frame3_1,frame3_2]...]
    return ((vxreal-vx)**2+(vyreal-vy)**2)**0.5

def calculVitMoyBetweenFr(annot, categories, last_ones, keys, keys2, times, a, b, giveZero=True):
    if keys2==[]:
        keys2=[k for k in last_ones]
    if times==[]:
        times={}
        for m in last_ones:
            times[m]=[keys[last_ones[m][0][0]],keys[last_ones[m][-1][0]]]
    sx=0
    sy=0
    l=0
    for m in last_ones:
        if times[m][0]>b or times[m][1]<a:
            continue
        goThrough=[k for k in last_ones[m] if keys[k[0]]>=a and keys[k[0]]<=b]
        if len(goThrough)<=1:
            continue
        goThrough.sort()
        
        i0=keys[goThrough[0][0]]
        j0=goThrough[0][1]
        (x0,y0)=calcCenter(annot[i0][j0][2:6])
        for k in range(1,len(goThrough)):
            i1=keys[goThrough[k][0]]
            j1=goThrough[k][1]
            (x1,y1)=calcCenter(annot[i1][j1][2:6])
            sx+=(x1-x0)/(i1-i0)
            sy+=(y1-y0)/(i1-i0)
            l+=1
            
    if l==0 and giveZero:
        return (0,0)
    elif l==0 and not giveZero:
        return (None,None)
    return (sx/l,sy/l)
        

def firstArgTuple(t):
    (a,b)=t
    return a

def lastArgTuple(t):
    (a,b)=t
    return b


    
def createMvmtWithVMoyCatNum(annot, categories, last_ones ,keys):
    l=0
    for m in last_ones:
        (Vx,Vy)=calculVitMoyCatNum(annot, categories, last_ones ,keys, m)
        #(Vx,Vy)=(0,0)
        #Vy=0
        l+=1
        if 'artVitCat' not in categories:
            categories.append('artVitCat')
        (a,b)=m
        iMin=mini([k[0] for k in last_ones[m]])
        j=[k[1] for k in last_ones[m]][argmin([k[0] for k in last_ones[m]])]
        coords=annot[keys[iMin]][j][2:6]
        iMax=-1*mini([-1*k[0] for k in last_ones[m]])
        for i in range(keys[iMin],keys[iMax]+1):
            if i not in annot:
                annot[i]=[]
            annot[i].append(['artVitCat',l,coords[0]+Vx*(i-keys[iMin]),coords[1]+Vy*(i-keys[iMin]),coords[2]+Vx*(i-keys[iMin]),coords[3]+Vy*(i-keys[iMin]),i])
    
    return (annot, categories)
    
def createMvmtWithVMoyCatNum2(annot, categories, last_ones ,keys):
    l=0
    for m in last_ones:
        (Vx,Vy)=calculVitMoyCatNum2(annot, categories, last_ones ,keys, m)
        #(Vx,Vy)=(0,0)
        #Vy=0
        l+=1
        if 'artVitCat' not in categories:
            categories.append('artVitCat')
        (a,b)=m
        iMin=mini([k[0] for k in last_ones[m]])
        j=[k[1] for k in last_ones[m]][argmin([k[0] for k in last_ones[m]])]
        coords=annot[keys[iMin]][j][2:6]
        iMax=-1*mini([-1*k[0] for k in last_ones[m]])
        for i in range(keys[iMin],keys[iMax]+1):
            if i not in annot:
                annot[i]=[]
            annot[i].append(['artVitCat',l,coords[0]+Vx*(i-keys[iMin]),coords[1]+Vy*(i-keys[iMin]),coords[2]+Vx*(i-keys[iMin]),coords[3]+Vy*(i-keys[iMin]),i])
    
    return (annot, categories)
    
def createMvmtWithV(annot, categories, last_ones ,keys, Vx, Vy):
    l=0
    for m in last_ones:
        l+=1
        if 'artVitMvmt' not in categories:
            categories.append('artVitMvmt')
        (a,b)=m
        iMin=mini([k[0] for k in last_ones[m]])
        j=[k[1] for k in last_ones[m]][argmin([k[0] for k in last_ones[m]])]
        coords=annot[keys[iMin]][j][2:6]
        iMax=-1*mini([-1*k[0] for k in last_ones[m]])
        for i in range(keys[iMin],keys[iMax]+1):
            if i not in annot:
                annot[i]=[]
            annot[i].append(['artVitMvmt',l,coords[0]+Vx*(i-keys[iMin]),coords[1]+Vy*(i-keys[iMin]),coords[2]+Vx*(i-keys[iMin]),coords[3]+Vy*(i-keys[iMin]),i])
    
    return (annot, categories)
        
def createMvmtWithPred(annot, categories, last_ones ,keys):
    l=0
    for m in last_ones:
        l+=1
        (a,b)=m
        if 'artPred_'+a not in categories:
            categories.append('artPred_'+a)
        iMin=mini([k[0] for k in last_ones[m]])
        jMin=[k[1] for k in last_ones[m]][argmin([k[0] for k in last_ones[m]])]
        coords=annot[keys[iMin]][jMin][2:6]
        iMax=-1*mini([-1*k[0] for k in last_ones[m]])
        jMax=[k[1] for k in last_ones[m]][argmin([-1*k[0] for k in last_ones[m]])]
        coordsMax=annot[keys[iMax]][jMax][2:6]
        num=annot[keys[iMax]][jMax][1]
        V1=(coordsMax[0]-coords[0])/(keys[iMax]-keys[iMin])
        V2=(coordsMax[1]-coords[1])/(keys[iMax]-keys[iMin])
        V3=(coordsMax[2]-coords[2])/(keys[iMax]-keys[iMin])
        V4=(coordsMax[3]-coords[3])/(keys[iMax]-keys[iMin])
        for i in range(keys[iMin],keys[iMax]+1):
            if i not in annot:
                annot[i]=[]
            annot[i].append(['artPred_'+a,num,coords[0]+V1*(i-keys[iMin]),coords[1]+V2*(i-keys[iMin]),coords[2]+V3*(i-keys[iMin]),coords[3]+V4*(i-keys[iMin]),i])

    return (annot, categories)
    
def FillInHoles(annot, categories, last_ones ,keys):
    l=0
    for m in last_ones:
        
        T=copy.copy(last_ones[m])
        T.sort()
        listeIndis=[keys[T[k][0]] for k in range(len(T))]
        indi=-1
        n=len(T)
        l+=1
        (a,b)=m
        if 'FillIn_'+a not in categories:
            categories.append('FillIn_'+a)
        iMin=mini([k[0] for k in last_ones[m]])
        jMin=[k[1] for k in last_ones[m]][argmin([k[0] for k in last_ones[m]])]
        coords=annot[keys[iMin]][jMin][2:6]
        iMax=-1*mini([-1*k[0] for k in last_ones[m]])
        jMax=[k[1] for k in last_ones[m]][argmin([-1*k[0] for k in last_ones[m]])]
        coordsMax=annot[keys[iMax]][jMax][2:6]
        num=annot[keys[iMax]][jMax][1]
        
        i0=0
        j0=0
        
        for i in range(keys[iMin],keys[iMax]+1):
            if i not in annot:
                annot[i]=[]
            if i in listeIndis and i!=keys[iMax]:
                
                indi+=1
                
                i0=T[indi][0]
                j0=T[indi][1]
                
                i1=T[indi+1][0]
                j1=T[indi+1][1]
                V1=(annot[keys[i1]][j1][2]-annot[keys[i0]][j0][2])/(keys[i1]-keys[i0])
                V2=(annot[keys[i1]][j1][3]-annot[keys[i0]][j0][3])/(keys[i1]-keys[i0])
                V3=(annot[keys[i1]][j1][4]-annot[keys[i0]][j0][4])/(keys[i1]-keys[i0])
                V4=(annot[keys[i1]][j1][5]-annot[keys[i0]][j0][5])/(keys[i1]-keys[i0])
                
            annot[i].append(['FillIn_'+a,num,annot[keys[i0]][j0][2]+V1*(i-keys[i0]),annot[keys[i0]][j0][3]+V2*(i-keys[i0]),annot[keys[i0]][j0][4]+V3*(i-keys[i0]),annot[keys[i0]][j0][5]+V4*(i-keys[i0]),i])

    return (annot, categories)
    
def calculVitMoyCatNum(annot, categories, last_ones ,keys, m):
    
    T=copy.copy(last_ones[m])
    T.sort()
    listeIndis=[keys[T[k][0]] for k in range(len(T))]
    n=len(T)
    (a,b)=m
    iMin=mini([k[0] for k in last_ones[m]])
    jMin=[k[1] for k in last_ones[m]][argmin([k[0] for k in last_ones[m]])]
    coords=annot[keys[iMin]][jMin][2:6]
    iMax=-1*mini([-1*k[0] for k in last_ones[m]])
    jMax=[k[1] for k in last_ones[m]][argmin([-1*k[0] for k in last_ones[m]])]
    coordsMax=annot[keys[iMax]][jMax][2:6]
    num=annot[keys[iMax]][jMax][1]
        
    i0=0
    j0=0
    sx=0
    sy=0
    for indi in range(n-1):
        i0=listeIndis[indi]
        j0=T[indi][1]

        i1=listeIndis[indi+1]
        j1=T[indi+1][1]
        
        (x0,y0)=calcCenter(annot[i0][j0][2:6])           
        (x1,y1)=calcCenter(annot[i1][j1][2:6])
                    
        sx+=(x1-x0)/(i1-i0)                      
        sy+=(y1-y0)/(i1-i0)           

    return (sx/(n-1),sy/(n-1))
    
def calculVitMoyCatNum2(annot, categories, last_ones ,keys, m):
    
    T=copy.copy(last_ones[m])
    T.sort()
    listeIndis=[keys[T[k][0]] for k in range(len(T))]
    n=len(T)
    (a,b)=m
    iMin=mini([k[0] for k in last_ones[m]])
    jMin=[k[1] for k in last_ones[m]][argmin([k[0] for k in last_ones[m]])]
    coords=annot[keys[iMin]][jMin][2:6]
    iMax=-1*mini([-1*k[0] for k in last_ones[m]])
    jMax=[k[1] for k in last_ones[m]][argmin([-1*k[0] for k in last_ones[m]])]
    coordsMax=annot[keys[iMax]][jMax][2:6]
    num=annot[keys[iMax]][jMax][1]
        
    i0=0
    j0=0
    sx=0
    sy=0
    for indi in range(n-1):
        i0=listeIndis[0]
        j0=T[0][1]

        i1=listeIndis[indi+1]
        j1=T[indi+1][1]
        
        (x0,y0)=calcCenter(annot[i0][j0][2:6])           
        (x1,y1)=calcCenter(annot[i1][j1][2:6])
                    
        sx+=(x1-x0)/(i1-i0)                   
        sy+=(y1-y0)/(i1-i0)           

    return (sx/(n-1),sy/(n-1))

        
def calculVit(annot, last_ones, keys):
    nbr=0
    sx=0
    sy=0
    for m in last_ones:
            #quadratique bcs little data
        if m not in [] :
            if len(last_ones[m])<=1:
                continue
            copyLastOne=copy.copy(last_ones[m])
            copyLastOne.sort()
            for l in range(1,len(last_ones[m])):
                
                vectA=annot[keys[copyLastOne[l-1][0]]][copyLastOne[l-1][1]]
                vectB=annot[keys[copyLastOne[l][0]]][copyLastOne[l][1]]
            #    print(m)
             #   print(l)
              #  print(vectA)
               # print(vectB)
                (cAx,cAy)=calcCenter(vectA[2:])
                (cBx,cBy)=calcCenter(vectB[2:])                                
                sx+=(cAx-cBx)/(vectA[6]-vectB[6])                       
                sy+=(cAy-cBy)/(vectA[6]-vectB[6])
                nbr+=1
    if nbr==0:
        return (0,0)
    return (sx/nbr,sy/nbr)
    
def calculVit2(annot, last_ones, keys):
    nbr=0
    sx=0
    sy=0
    for m in last_ones:
            #quadratique bcs little data
        if m not in [] :
            if len(last_ones[m])<=1:
                continue
            copyLastOne=copy.copy(last_ones[m])
            copyLastOne.sort()
            for l in range(1,len(last_ones[m])):
                
                vectA=annot[keys[copyLastOne[0][0]]][copyLastOne[0][1]]
                vectB=annot[keys[copyLastOne[l][0]]][copyLastOne[l][1]]
            #    print(m)
             #   print(l)
              #  print(vectA)
               # print(vectB)
                (cAx,cAy)=calcCenter(vectA[2:])
                (cBx,cBy)=calcCenter(vectB[2:])                                
                sx+=(cAx-cBx)/(vectA[6]-vectB[6])                       
                sy+=(cAy-cBy)/(vectA[6]-vectB[6])
                nbr+=1
    if nbr==0:
        return (0,0)
    return (sx/nbr,sy/nbr)
    
def calcCenter(vectA):
    return ((vectA[2]+vectA[0])/2,(vectA[1]+vectA[3]))

def decale(vect,vite):
    ta=vect[:]
    ta[0]+=vite[0]
    ta[2]+=vite[0]
    
    ta[1]+=vite[1]
    ta[3]+=vite[1]
    return ta
    
def CreateForVeng(annot, categories, last_ones ,keys):
    name=path
    name = name.split('.')[0]+'_veng.txt'
    f=open(name,'w+')
    for m in last_ones:
        (a,b)=m
        vect=annot[keys[last_ones[m][0][0]]][last_ones[m][0][1]]
        f.write(str(vect[2])+' '+str(vect[3])+' '+str(vect[4])+' '+str(vect[5])+' '+str(vect[6])+' '+str(a)+' '+str(b)+'\n')
    f.close()
    
def ridOfSameCatSameF(annot, categories, last_ones ,keys):
    keys3=annot.keys()
    keys3=[k for k in keys3]
    keys3.sort()
    

#dumbCalcul()
p1=[0,10,20,100]
p2=[0,20,10,30]
alpha=2
alpha_vit=1.5
seuilToRid=0.15
seuilLeave=0.17
beta=1
limMaxReappear=10
gamma=300
dumbCalculWithDist2(gamma/((beta)**0.5),(fps*2)//3,3,[0,0,'begin']) #2/3 282.85 10/8/6*
