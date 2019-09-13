
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
UBIT ='manishre'
np.random.seed(sum([ord(c) for c in UBIT]))


# In[2]:


def epipolarLines(left_image,right_image,lines,src_pts,des_pts):
    r,c = left_image.shape[:2]
    
    for r,pt1,pt2 in zip(lines,src_pts,des_pts):
        color = tuple(np.random.randint(0,255,3).tolist()) # picking random color for each line 
        x0,y0 = map(int, [0, -r[2]/r[1] ]) # 
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        left_image = cv2.line(left_image, (x0,y0), (x1,y1), color,1)
        left_image = cv2.circle(left_image,tuple(pt1),5,color,-1)
        right_image = cv2.circle(right_image,tuple(pt2),5,color,-1)

    return left_image,right_image


# In[3]:


image1 = cv2.imread('tsucuba_left.png') 
image2 = cv2.imread('tsucuba_right.png')
image3 = cv2.imread('tsucuba_left.png',0)
image4 =cv2. imread('tsucuba_right.png',0)


# In[4]:


sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
detected_points1, descriptors1 = sift.detectAndCompute(image1.copy(),None)
detected_points2, descriptors2 = sift.detectAndCompute(image2.copy(),None)
detect_image1 =cv2.drawKeypoints(image1,detected_points1,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
detect_image2 = cv2.drawKeypoints(image2,detected_points2,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(descriptors1,descriptors2,k=2)

good = [] 
good_pts =[]
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.75*n.distance:
        good.append(m)
        good_pts.append([m])
        pts2.append(detected_points2[m.trainIdx].pt)
        pts1.append(detected_points1[m.queryIdx].pt)



featureMacthing = cv2.drawMatchesKnn(image1.copy(),detected_points1,image2.copy(),detected_points2,good_pts,None,flags=2)
#======================================PART2=====================================================================

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

fundamentalMatrix, mask = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC)
print('===FUNDAMENTAL MATRIX===')
print(fundamentalMatrix)
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]
print(len(pts1))
key=[]
for i in range(10):
     key.append(random.randint(1,100))
print('key',key)
pts3 =[]
pts4 =[]
for i in key:
    pts3.append(pts1[i])
    pts4.append(pts2[i])
    
pts3 =np.asarray(pts3)
pts4 =np.asarray(pts4)

#============================================Part3=========================================================================
inliners_left = cv2.computeCorrespondEpilines(pts3.reshape(-1,1,2), 2,fundamentalMatrix)
inliners_left = inliners_left.reshape(-1,3)
image_l2r,image_l2rp = epipolarLines(image1.copy(),image2.copy(),inliners_left,pts3,pts4)

inliners_right =cv2.computeCorrespondEpilines(pts4.reshape(-1,1,2),2,fundamentalMatrix)
inliners_right = inliners_right.reshape(-1,3)
image_r2l,image_r2lp = epipolarLines(image2.copy(),image1.copy(),inliners_right,pts3,pts4)



#==============================================part4========================================================================

depthMap = cv2.StereoBM_create(numDisparities=64, blockSize=21)

#depthMap = cv2.createStereoBM(numDisparities=64, blockSize=21)
__disparityMap = depthMap.compute(image3,image4)
__disparityMap = (__disparityMap,0)





# In[ ]:


cv2.imshow('keypoint1',detect_image1)
cv2.waitKey(0)
cv2.imshow('keypoints2',detect_image2)
cv2.waitKey(0)
cv2.imshow('Feature Matching',featureMacthing)
cv2.waitKey(0)

cv2.imshow('epipolarLines from Left to right',image_l2r)
cv2.waitKey(0)
cv2.imshow('epipolar points from left to right',image_l2rp)
cv2.waitKey(0)
cv2.imshow('epipolarlines from right to left',image_r2l)
cv2.waitKey(0)
cv2.imshow('epipolar points from right to left',image_r2lp)
cv2.waitKey(0)
plt.imshow(__disparityMap,'gray')
cv2.destroyAllWindows()
#print(fundamental)


# In[ ]:


cv2.imwrite('task2_sift1.jpg',detect_image1)
cv2.imwrite('task2_sift2.jpg',detect_image2)
cv2.imwrite('task2_matches_knn.jpg',featureMacthing)
cv2.imwrite('task2_epi_right.jpg',image_l2r)
cv2.imwrite('task2_epi_left.jpg',image_r2l)
cv2.imwrite('task2_disparity.jpg',__disparityMap)

