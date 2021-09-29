# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 15:36:47 2021

@author: Priam CARDOUAT
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
rho=0.3

def load_movielens(filename, minidata=False):
    """
    Cette fonction lit le fichier filename de la base de donnees
    Movielens, par exemple 
    filename = '~/datasets/ml-100k/u.data'
    Elle retourne 
    R : une matrice utilisateur-item contenant les scores
    mask : une matrice valant 1 si il y a un score et 0 sinon
    """

    data = np.loadtxt(filename, dtype=int)

    R = sparse.coo_matrix((data[:, 2], (data[:, 0]-1, data[:, 1]-1)),
                          dtype=float)
    R = R.toarray()  # not optimized for big data

    # code la fonction 1_K
    mask = sparse.coo_matrix((np.ones(data[:, 2].shape),
                              (data[:, 0]-1, data[:, 1]-1)), dtype=bool )
    mask = mask.toarray()  # not optimized for big data

    if minidata is True:
        R = R[0:100, 0:200].copy()
        mask = mask[0:100, 0:200].copy()

    return R, mask

print(load_movielens('u.data'))
R=load_movielens('u.data')[0]
mask=load_movielens('u.data')[1]

Q0, _, P0 = svds(R,k=4)

def total_objective(P, Q, R, mask, rho):
    """
    La fonction objectif du probleme complet.
    Prend en entree 
    P : la variable matricielle de taille C x I
    Q : la variable matricielle de taille U x C
    R : une matrice de taille U x I
    mask : une matrice 0-1 de taille U x I
    rho : un reel positif ou nul

    Sorties :
    val : la valeur de la fonction
    grad_P : le gradient par rapport a P
    grad_Q : le gradient par rapport a Q
    """

    tmp = (R - Q.dot(P)) * mask

    val = np.sum(tmp ** 2)/2. + rho/2. * (np.sum(Q ** 2) + np.sum(P ** 2))

    grad_P = np.transpose(Q).dot((Q.dot(P)-R)*mask) + rho*P

    grad_Q =((Q.dot(P)-R)*mask).dot(np.transpose(P)) + rho*Q

    return val, grad_P, grad_Q

def total_objective_vectorized(PQvec, R, mask, rho):
    """
    Vectorisation de la fonction precedente de maniere a ne pas
    recoder la fonction gradient
    """

    # reconstruction de P et Q
    n_items = R.shape[1]
    n_users = R.shape[0]
    F = PQvec.shape[0] // (n_items + n_users)
    Pvec = PQvec[0:n_items*F]
    Qvec = PQvec[n_items*F:]
    P = np.reshape(Pvec, (F, n_items))
    Q = np.reshape(Qvec, (n_users, F))

    val, grad_P, grad_Q = total_objective(P, Q, R, mask, rho)
    return val, np.concatenate([grad_P.ravel(), grad_Q.ravel()])

def objective(P, Q0, R, mask, rho):
    """
    La fonction objectif du probleme simplifie.
    Prend en entree 
    P : la variable matricielle de taille C x I
    Q0 : une matrice de taille U x C
    R : une matrice de taille U x I
    mask : une matrice 0-1 de taille U x I
    rho : un reel positif ou nul

    Sorties :
    val : la valeur de la fonction
    grad_P : le gradient par rapport a P
    """

    tmp = (R - Q0.dot(P)) * mask

    val = np.sum(tmp ** 2)/2. + rho/2. * (np.sum(Q0 ** 2) + np.sum(P ** 2))

    grad_P = np.transpose(Q0).dot((Q0.dot(P)-R)*mask) + rho*P

    return val, grad_P


def gradient(g, P0, Q0, epsilon,a,b):
    Pk=P0
    Qk=Q0
    while np.sum(total_objective(Pk,Qk,R,mask,rho)[1]**2)+np.sum(total_objective(Pk,Qk,R,mask,rho)[2]**2)>epsilon**2:
        l=0
        while (total_objective(Pk-(b*(a**l))*total_objective(Pk,Qk,R,mask,rho)[1],Qk-(b*(a**l))*total_objective(Pk,Qk,R,mask,rho)[2],R,mask,rho)[0])>(total_objective(Pk,Qk,R,mask,rho)[0]+np.trace(np.dot(total_objective(Pk,Qk,R,mask,rho)[1],np.transpose(Pk-(b*(a**l))*total_objective(Pk,Qk,R,mask,rho)[1]-Pk)))+np.trace(np.dot(total_objective(Pk,Qk,R,mask,rho)[2],np.transpose(Qk-(b*(a**l))*total_objective(Pk,Qk,R,mask,rho)[2]-Qk)))+(1/(2*b*(a**l)))*(np.sum((Pk-Pk-(b*(a**l))*total_objective(Pk,Qk,R,mask,rho)[1])**2)+np.sum((Qk-Qk-(b*(a**l))*total_objective(Pk,Qk,R,mask,rho)[2])**2))):
            l+=1
        Pk=Pk-(b*(a**l))*total_objective(Pk,Qk,R,mask,rho)[1]
        Qk=Qk-(b*(a**l))*total_objective(Pk,Qk,R,mask,rho)[2]
    return (Pk,Qk)



P,Q=gradient(_,P0,Q0,100,0.5,0.01)
R_bis=np.dot(Q,P)
maximum=0
movie=0
for k in range(len(R_bis[0])):
    if R_bis[300][k]>=maximum:
        maximum=R_bis[300][k]
        movie=k
print(movie)