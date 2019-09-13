 Project 2
==========================
Manish Reddy Challamala, November 5, 2018, manishre@buffalo.edu

For detailed explaination, please visit the below link:
[link for report pdf](https://github.com/manish216/CSE-573-Project2/blob/master/CVIP_project_2.pdf)

## 1 Task1 - Image Features and Homography

## Abstract
The goal of this task is stated point wise below:

1. To find the SIFT feature between two images and plot the key points in the images.
2. Compute all the k-nearest neighbours according to the given condition stated in the task and draw all the matches.
3. Compute the Homo graphic Matrix using RANSAC.
4. Plot 10 random matches using only inliners.
5. wrap the image 2 with image 1 using computed homography matrix.

## 1.1 Experimental set-up:

1. For this task, we are using the following libraries:
    1. Cv2, numpy, matplotlib
2. Firstly, calculating the keypoints and descriptors of the given two images
    mountain1.jpg ans mountain2.jpg and plotting the respective key-points on the image.
3. For the task 1.2 calculating the matches using the FlannBasedMatcher function by sending
   the calculated keypoints of both images as input.
4. Plotting all the macthes found between two images by using drawMatchesKnn built in fucntion ofcv2 library.
5. Computing the homography matrix and displaying the results on the screen.
6. selecting 10 random macthes using inliners and plotting them along the images.
7. Finally, wrapping the image by first transforming the prespective of the image2 and 
   aligning it according to the image 1 and merging the both images together by decreasing the distance between them.


## 2 Task 2:


### Abstract
The goal of this task is stated point wise below:

1. To find the SIFT feature between two images and plot the key points in the images ans also Compute all the k-nearest neighbours according 
    to the given condition stated in the task and draw all the matches.
2. Compute the Fundamental Matrix using RANSAC.
3. Plot 10 random matches using only inliners. and for each keypoint in right compute and draw the epiline on the left image. and vice-versa
4. Compute the disparity map for both left and right images.

### 2.1 Experimental set-up:

1. For this task, we are using the following libraries:
    1. Cv2, numpy, matplotlib
2. Firstly, calculating the keypoints and descriptors of the given two images
    tsucubaleft.png ans tsucubaleft.png and plotting the respective key-points on the image.
3. For the task 2.1 calculating the matches using the FlannBasedMatcher function by sending the calculated keypoints of both images as input.
4. Plotting all the macthes found between two images by using drawMatchesKnn built in fucntion ofcv2 library.
5. Computing the Fundamental Matrix and displaying the results on thescreen.
6. selecting 10 random macthes using inliners and for each keypoint compute the epiline and plot the line with respect o the image.
7. Compute the disparity map:
    1.Create stereoBMcreate object with numDisparites =64 and block size =21 has parameters.
    2. use this object and compute the disparity map for two given images


## 3 Task

### Abstract
The goal of this task is stated point wise below:
1. Compute the classification vector and plot of given N samples and corresponding centroids.
2. Recompute centroid value and plot the graph of corresponding cetroids
3. Repeat the above process for 2nd iteration and plot the graph

### 3.1 Experimental setup:

1. calculate the euclidean distance and update the cluster points.
2. cluster the points according to there nearest centroid.
3. plot the graph of the new distance points.
4. recompute the centroid values by taking the average of new distance value of x and y.
5. plot the graph for new centroid values.
6. repeat the above steps for seconf iteration.

