
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


# In[86]:


def _clustering(a, b):
    return np.linalg.norm(a - b, axis=1)


# In[179]:


x_1 = [5.9,4.6,6.2,4.7,5.5,5.0,4.9,6.7,5.1,6.0]
y_1 = [3.2,2.9,2.8,3.2,4.2,3.0,3.1,3.1,3.8,3.0]
X = np.asarray(list(zip(x_1,y_1)))
mu_x =[6.2,6.6,6.5]
mu_y =[3.2,3.7,3.0]
mu = np.asarray(list(zip(mu_x,mu_y)))
print(mu)
new_mu = np.zeros(len(X))
for i in range(len(X)):
    new_samples = _clustering(X[i], mu)
    c =  np.argmin(new_samples)
    new_mu[i] = c
print(new_mu)
new_samples =[]
new_x =[]
colors = ['r', 'g', 'b']
figure1, axis1 = plt.subplots()
for k in range(3):
    temp = np.array([X[i] for i in range(len(X)) if new_mu[i] == k])
    new_x.append(temp)
    print("cluster" + str(k+1), temp )
    axis1.scatter(temp[:,0], temp[:,1], marker='^', s=90,facecolor ='#FFFFFF' ,edgecolor=colors)
axis1.scatter(mu[:,0], mu[:,1], marker='o', s=50, facecolor=colors)
figure1.savefig('task3_iter1_a.jpg', dpi=500)


# In[173]:


c1 =np.asarray(new_x[0])
c2 =np.asarray(new_x[1])
c3 =np.asarray(new_x[2])
c1_xaxis =c1[:,0]
c1_yaxis =c1[:,1]
c2_xaxis =c2[:,0]
c2_yaxis =c2[:,1]
c3_xaxis =c3[:,0]
c3_yaxis =c3[:,1]
#========= Part2====================
mu_1 = [(sum(c1_xaxis)+mu[0,0])/(len(c1_xaxis)+1),(sum(c1_yaxis)+mu[0,0])/(len(c1_yaxis)+1)]
mu_2 = [(sum(c2_xaxis)+mu[0,0])/(len(c2_xaxis)+1),(sum(c2_yaxis)+mu[0,0])/(len(c2_yaxis)+1)]
mu_3 = [(sum(c3_xaxis)+mu[0,0])/(len(c3_xaxis)+1),(sum(c3_yaxis)+mu[0,0])/(len(c3_yaxis)+1)]
print(mu_1)
print(mu_2)
print(mu_3)
figure2, axis2 = plt.subplots()
mu_x1 =[mu_1[0],mu_2[0],mu_3[0]]
mu_y1 =[mu_1[1],mu_2[1],mu_3[1]]
NewMu = np.asarray(list(zip(mu_x1,mu_y1)))
color =['r','b','g']
axis2.scatter(mu_x1, mu_y1, marker='o', s=50, c=color)
print("newmu",NewMu)
figure2.savefig('task3_iter1_b.jpg', dpi=500)


# In[190]:


#=========part3=====================
X1 = np.asarray(list(zip(x_1,y_1)))
new_mu1 = np.zeros(len(X1))
for i in range(len(X1)):
    new_samples1 = _clustering(X1[i], NewMu)
    c1 =  np.argmin(new_samples1)
    new_mu1[i] = c1
new_x1 =[]
colors = ['r', 'g', 'b']
figure3, axis3 = plt.subplots()
for k in range(3):
    temp1 = np.array([X1[i] for i in range(len(X1)) if new_mu1[i] == k])
    new_x1.append(temp1)
    print("cluster" + str(k+1), temp )
    axis3.scatter(temp1[:,0], temp1[:,1], marker='^', s=90,facecolor ='#FFFFFF' ,edgecolor=colors)


# In[175]:


c1 =np.asarray(new_x[0])
c2 =np.asarray(new_x[1])
c3 =np.asarray(new_x[2])
c1_xaxis =c1[:,0]
c1_yaxis =c1[:,1]
c2_xaxis =c2[:,0]
c2_yaxis =c2[:,1]
c3_xaxis =c3[:,0]
c3_yaxis =c3[:,1]
#========= Part2====================
mu_1 = [(sum(c1_xaxis)+mu[0,0])/(len(c1_xaxis)+1),(sum(c1_yaxis)+mu[0,0])/(len(c1_yaxis)+1)]
mu_2 = [(sum(c2_xaxis)+mu[0,0])/(len(c2_xaxis)+1),(sum(c2_yaxis)+mu[0,0])/(len(c2_yaxis)+1)]
mu_3 = [(sum(c3_xaxis)+mu[0,0])/(len(c3_xaxis)+1),(sum(c3_yaxis)+mu[0,0])/(len(c3_yaxis)+1)]
print(mu_1)
print(mu_2)
print(mu_3)
figure2, axis2 = plt.subplots()
mu_x1 =[mu_1[0],mu_2[0],mu_3[0]]
mu_y1 =[mu_1[1],mu_2[1],mu_3[1]]
NewMu = np.asarray(list(zip(mu_x1,mu_y1)))
color =['r','b','g']
axis2.scatter(mu_x1, mu_y1, marker='o', s=50, c=color)
print("newmu",NewMu)
figure2.savefig('task3_iter1_b.jpg', dpi=500)

