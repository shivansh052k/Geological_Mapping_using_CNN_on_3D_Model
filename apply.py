# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:57:44 2021

@author: Moi
"""
from visualize import load_image2torch, aspp_output, get_alphas, \
    load_trained_model, hclustering_output, kclustering_output, return_acmap
import numpy as np
import torch
import gc
from os import remove
import skimage.transform

def data2torch(data):
    """
    fonction pour changer les données de data à un format compatible pour
    les réseaux
    """
    data_temp = []
    for i in data:
        i = load_image2torch(i)
        data_temp.append(i)
        
    return data_temp

def assp_all(data, net, path):
    """
    fonction pour faire le assp sur l'ensemble de la zone
    in : data - les données au format torch,
         net - le modèle chargée avec les poids
         path - l'emplacement ou enregistrer le fichier en .npy
    out : DANS UN FICHIER .NPY im path
          [assp - liste des assp pour chaque données
          seg - liste des seg pour 'ensemble des donéées
          edge - l'ensemble des edges pour les données]
    """
    
    try :
        remove(path+'_assp.npy')
    except FileNotFoundError:
        pass
    
    try :
        remove(path+'_seg.npy')
    except FileNotFoundError:
        pass
    
    try :
        remove(path+'_edge.npy')
    except FileNotFoundError:
        pass
    
    f1=open(path+'_assp.npy','ab')
    f2=open(path+'_seg.npy','ab')
    f3=open(path+'_edge.npy','ab')
    
    for i in data:
        asppt, segt, edget = aspp_output(net, i)

        #enregistrer dans un fichier
        np.save(f1, asppt.cpu().numpy()[0])
        np.save(f2, segt.max(1)[1][0].cpu().numpy())
        np.save(f3, edget.max(1)[0][0].cpu().numpy())
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
     
    f1.close() 
    f2.close() 
    f3.close() 
    
def gate_all(data, net, path):
    """
    fonction pour faire le assp sur l'ensemble de la zone
    in : data - les données au format torch,
         net - le modèle chargée avec les poids
         path - l'emplacement ou enregistrer le fichier en .npy
    out : DANS UN FICHIER .NPY im path
          [assp - liste des assp pour chaque données
          seg - liste des seg pour 'ensemble des donéées
          edge - l'ensemble des edges pour les données]
    """
    
    try :
        remove(path+'_gate1.npy')
    except FileNotFoundError:
        pass
    
    try :
        remove(path+'_gate2.npy')
    except FileNotFoundError:
        pass
    
    try :
        remove(path+'_gate3.npy')
    except FileNotFoundError:
        pass
    
    f1=open(path+'_gate1.npy','ab')
    f2=open(path+'_gate2.npy','ab')
    f3=open(path+'_gate3.npy','ab')
    
    for i in data:
        dsn3, dsn4, dsn7 = get_alphas(net, i)

        #enregistrer dans un fichier
        np.save(f1, dsn3.cpu().numpy()[0, 0, :, :])
        np.save(f2, dsn4.cpu().numpy()[0, 0, :, :])
        np.save(f3, dsn7.cpu().numpy()[0, 0, :, :])
        
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
     
    f1.close() 
    f2.close() 
    f3.close() 
            
def load_npy(path):
    """
    Fonction pour ouvrir les fichier .npy pour visualisation
    in : path - l'emplacement des données .npy
    out : data - les données chargées
    """
    f =open(path, 'rb')
    data_temp = []
    while True:
        try :
            data_temp.append(np.load(f))
        except ValueError:
            break
         
    f.close()
    data_temp = np.array(data_temp)

    return data_temp

def con_im(data_org1, li, al=42):
    
    data_org = data_org1.copy()
    temp = data_org[0][al:-al,al:-al]
    temp[:] = np.nan
        
    
    for l in li:
        a=l[0]
        b=l[1]
        c1, c2 = l[2]
        
        temp11 = temp
        for i in range(c1):
            temp11 = np.hstack((temp11, temp))
        
        temp12 = temp
        for i in range(c2):
            temp12 = np.hstack((temp12, temp))
            
        ini = np.hstack((temp11, data_org[a][al:-al, al:-al]))
                        
        for i in range(a+1, b):
            ini = np.hstack((ini, data_org[i][al:-al, al:-al]))
            
        ini = np.hstack((ini, temp12))

        if 'ini_t' not in locals(): #locals
            ini_t = ini
        else : 
            ini_t = np.vstack((ini_t, ini))
            
    return ini_t

def resize_image(im, rat):
    x, y = im.shape
    x, y  = int(x*rat), int(y*rat)
    
    return skimage.transform.resize(im, (x,y), order=3)