
# coding: utf-8

# In[1]:


import cv2
import random
UBIT ='manishre'
import numpy as np
np.random.seed(sum([ord(c) for c in UBIT]))
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


# ### Reading the Images

# In[2]:


image1 = cv2.imread('mountain1.jpg')
image2 = cv2.imread('mountain2.jpg')


# In[3]:


# SIFT OBJECT CREATION
sift = cv2.xfeatures2d.SIFT_create()
# CALUCLATING the keypoints and descriptors
detected_points1, descriptors1 = sift.detectAndCompute(image1,None)
detected_points2, descriptors2 = sift.detectAndCompute(image2,None)
# displaying the keypoints of the images
detect_image1 =cv2.drawKeypoints(image1,detected_points1,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
detect_image2 = cv2.drawKeypoints(image2,detected_points2,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#=====================================part2===========================================================================
# Feature Matching 
Flann_index = 0
index_parameters = dict(algorithm = Flann_index, trees = 5)
search_parameters = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_parameters,search_parameters)

matches = flann.knnMatch(descriptors1,descriptors2,k=2)

matchesMask = [[0,0] for i in range(len(matches))]
#Apply ratio test
good = []
good_pts =[]
for i,(m,n) in enumerate(matches):
    if m.distance < 0.75*n.distance:
        good.append([m])
        good_pts.append(m)


featureMacthing = cv2.drawMatchesKnn(image1,detected_points1,image2,detected_points2,good,None,flags=2)

#=======================================part3==============================================================================
src_pts = np.float32([ detected_points1[m.queryIdx].pt for m in good_pts ]).reshape(-1,1,2)
dst_pts = np.float32([ detected_points2[m.trainIdx].pt for m in good_pts ]).reshape(-1,1,2)
print(src_pts.shape)
print(dst_pts.shape)
homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
#print(mask)
#=====================================part4================================================================================
inliner =[]
for i in range(len(mask)):
    if(mask[i]==1):
        inliner.append(good[i])
        #print(i)

wrappi
__10match = [] 
    
__10match = random.sample(inliner,10)


__10inlinerMatch = cv2.drawMatchesKnn(image1,detected_points1,image2,detected_points2,__10match,None,(255,0,0),flags=2)

#====================================part5===================================================================================

row1 = image1.shape[0]
row2 = image2.shape[0]
col1 = image1.shape[1]
col2 = image2.shape[1]

points_1 = np.float32([[0,0], [0,row1], [col1, row1], [col1,0]]).reshape(-1,1,2)
temp = np.float32([[0,0], [0,row2], [col2, row2], [col2,0]]).reshape(-1,1,2)

points_2 = cv2.perspectiveTransform(temp, homography)
points = np.concatenate((points_1, points_2), axis=0)

[x_min, y_min] = np.int32(points.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(points.max(axis=0).ravel() + 0.5)

translation_dist = [-x_min, -y_min]
H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

output_img = cv2.warpPerspective(image1, H_translation.dot(homography), (x_max - x_min, y_max - y_min))
output_img[translation_dist[1]:row1+translation_dist[1],translation_dist[0]:col1+translation_dist[0]] = image2


wrappedImage = cv2.warpPerspective(image1, homography, (image2.shape[1],image2.shape[0]))
     


# In[4]:


cv2.imshow('keypoint1',detect_image1)
cv2.waitKey(0)
cv2.imshow('keypoints2',detect_image2)
cv2.waitKey(0)
cv2.imshow('Feature Matching',featureMacthing)
cv2.waitKey(0)
cv2.imshow('10 random inliner matching',__10inlinerMatch)
cv2.waitKey(0)
cv2.imshow('wrap',output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("======HOMOGRAPHY MATRIX=====")
print(homography)


# In[5]:


cv2.imwrite('task1_sift1.jpg',detect_image1)
cv2.imwrite('task1_sift2.jpg',detect_image2)
cv2.imwrite('task1_matches_knn.jpg',featureMacthing)
cv2.imwrite('task1_matches.jpg',__10inlinerMatch)
cv2.imwrite('task1_pano.jpg',output_img)

