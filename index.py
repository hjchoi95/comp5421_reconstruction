import numpy as np
from numpy import matrix
from numpy import linalg
import math
import cv2 as cv
print "opencv version=",cv.__version__
from matplotlib import pyplot as plt
from utils import *

####################################################
# PART 1: Feature Detection
####################################################

print "=============Feature Detection=============="
#initiate images
img1 = cv.imread("images/0005.png")
img1 = cv.resize(img1, (0,0), fx=0.3, fy=0.3)
img2 = cv.imread("images/0006.png")
img2 = cv.resize(img2, (0,0), fx=0.3, fy=0.3)


# compute harris keypoints and plots
h_img1 = cv.imread("images/0005.png")
h_img1 = cv.resize(h_img1, (0,0), fx=0.3, fy=0.3)
gray1 = cv.cvtColor(h_img1,cv.COLOR_BGR2GRAY)
gray1 = np.float32(gray1)
h_img2 = cv.imread("images/0006.png")
h_img2 = cv.resize(h_img2, (0,0), fx=0.3, fy=0.3)
gray2 = cv.cvtColor(h_img2,cv.COLOR_BGR2GRAY)
gray2 = np.float32(gray2)

dst1 = cv.cornerHarris(gray1,2,3,0.04)
dst1 = cv.dilate(dst1,None)
dst2 = cv.cornerHarris(gray2,2,3,0.04)
dst2 = cv.dilate(dst2,None)
# Threshold for an optimal value, it may vary depending on the image.
h_img1[dst1>0.01*dst1.max()]=[0,0,255]
h_img2[dst2>0.01*dst2.max()]=[0,0,255]
plt.figure().canvas.set_window_title("harris detector")
print "close the plot window to proceed..."
plt.subplot(121),plt.imshow(h_img1), plt.subplot(122),plt.imshow(h_img2), plt.show()

# Initiate SIFT and SURF detector
sift = cv.xfeatures2d.SIFT_create(400,3,0.05,10,2.4)
surf = cv.xfeatures2d.SURF_create(800)

# compute SIFT keypoints
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
print "img1: SIFT descriptors =","count =",des1.size/des1[0].size,"data dimension =",des1[0].size
print "img2: SIFT descriptors =","count =",des2.size/des2[0].size,"data dimension =",des2[0].size

# compute SURF keypoints
kp1_surf, des1_surf = surf.detectAndCompute(img1,None)
kp2_surf, des2_surf = surf.detectAndCompute(img2,None)
print "img1: SURF descriptors =","count =",des1_surf.size/des1_surf[0].size,"data dimension =",des1_surf[0].size
print "img2: SURF descriptors =","count =",des2_surf.size/des2[0].size,"data dimension =",des2_surf[0].size

plt.figure().canvas.set_window_title("SIFT detector")
plt.subplot(121),plt.imshow(cv.drawKeypoints(img1,kp1,None))
plt.subplot(122),plt.imshow(cv.drawKeypoints(img2,kp2,None)), plt.show()

plt.figure().canvas.set_window_title("SURF detector")
plt.subplot(121),plt.imshow(cv.drawKeypoints(img1,kp1_surf,None))
plt.subplot(122),plt.imshow(cv.drawKeypoints(img2,kp2_surf,None)), plt.show()

####################################################
# PART 2: Feature Matching
####################################################

print "=============Feature Matching=============="
#implementation from scratch
print "computing matches using nearest neighbor method..."
nnMatches = findNNMatch(des1,des2,ratioTest=0.8)
print "nnMatch result: n/n' ratio test filtered out",des2.size/des2[0].size-len(nnMatches),"items from",des2.size/des2[0].size,"matches"
plt.figure().canvas.set_window_title("feature matches")
print "close the plot window to proceed..."
plt.imshow(cv.drawMatches(img1,kp1,img2,kp2,nnMatches, None, flags=2)),plt.show()

nnMatches = sorted(nnMatches, key = lambda x:x.distance)
nnMatches = nnMatches[:15]
plt.figure().canvas.set_window_title("best 15 feature matches")
print "close the plot window to proceed..."
plt.imshow(cv.drawMatches(img1,kp1,img2,kp2,nnMatches, None, flags=2)),plt.show()

####################################################
# PART 3: Epipolar geometry
####################################################

print "=============Epipolar geometry=============="
points1 = []
points2 = []
for m in nnMatches:
    points1.append(kp1[m.queryIdx].pt)
    points2.append(kp2[m.trainIdx].pt)

points1 = np.int32(points1)
points2 = np.int32(points2)
#compute 8-point algorithm
#print points1
#print points2
F,_ = cv.findFundamentalMat(points1,points2,cv.FM_8POINT)
#print F
print "Fundamental matrix using 8-point algorithm:"
computeFundamentalMatrix(points1, points2, F)
print F
print "computing epilines.."
# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(points2.reshape(-1,1,2), 2, F)
lines1 = lines1.reshape(-1,3)
lines2 = cv.computeCorrespondEpilines(points1.reshape(-1,1,2), 1, F)
lines2 = lines2.reshape(-1,3)
#print lines1
plt.figure().canvas.set_window_title("epilines from 8-point algorithm")
plt.subplot(121),plt.imshow(drawlines(img1.copy(),lines1,points1,points2))
plt.subplot(122),plt.imshow(drawlines(img2.copy(),lines2,points2,points1))
print "close the plot window to proceed..."
plt.show()

#7Point algorithm
print "Three Fundamental matrices using 7-point algorithm:"
F,_ = cv.findFundamentalMat(points1[:7],points2[:7],cv.FM_7POINT)
print F[0:3],","
print F[3:6],","
print F[6:9]
print "computing epilines.."
lines1 = cv.computeCorrespondEpilines(points2.reshape(-1,1,2), 2, F[0:3])
lines1 = lines1.reshape(-1,3)
lines2 = cv.computeCorrespondEpilines(points1.reshape(-1,1,2), 1, F[0:3])
lines2 = lines2.reshape(-1,3)
plt.figure().canvas.set_window_title("first solution for 7-point algorithm")
plt.subplot(121),plt.imshow(drawlines(img1.copy(),lines1,points1,points2))
plt.subplot(122),plt.imshow(drawlines(img2.copy(),lines2,points2,points1))
print "close the plot window to proceed..."
plt.show()
if F[3:6].size!=0:
    lines1 = cv.computeCorrespondEpilines(points2.reshape(-1,1,2), 2, F[3:6])
    lines1 = lines1.reshape(-1,3)
    lines2 = cv.computeCorrespondEpilines(points1.reshape(-1,1,2), 1, F[3:6])
    lines2 = lines2.reshape(-1,3)
    plt.figure().canvas.set_window_title("second solution for 7-point algorithm")
    plt.subplot(121),plt.imshow(drawlines(img1.copy(),lines1,points1,points2))
    plt.subplot(122),plt.imshow(drawlines(img2.copy(),lines2,points2,points1))
    print "close the plot window to proceed..."
    plt.show()
    lines1 = cv.computeCorrespondEpilines(points2.reshape(-1,1,2), 2, F[6:9])
    lines1 = lines1.reshape(-1,3)
    lines2 = cv.computeCorrespondEpilines(points1.reshape(-1,1,2), 1, F[6:9])
    lines2 = lines2.reshape(-1,3)
    plt.figure().canvas.set_window_title("third solution for 7-point algorithm")
    plt.subplot(121),plt.imshow(drawlines(img1.copy(),lines1,points1,points2))
    plt.subplot(122),plt.imshow(drawlines(img2.copy(),lines2,points2,points1))
    print "close the plot window to proceed..."
    plt.show()

#RANSAC algorithm
F,_ = cv.findFundamentalMat(points1,points2,cv.FM_RANSAC)
print F
print "computing epilines.."
# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(points2.reshape(-1,1,2), 2, F)
lines1 = lines1.reshape(-1,3)
lines2 = cv.computeCorrespondEpilines(points1.reshape(-1,1,2), 1, F)
lines2 = lines2.reshape(-1,3)
#print lines1
plt.figure().canvas.set_window_title("epilines from RANSAC algorithm")
plt.subplot(121),plt.imshow(drawlines(img1.copy(),lines1,points1,points2))
plt.subplot(122),plt.imshow(drawlines(img2.copy(),lines2,points2,points1))
print "close the plot window to proceed..."
plt.show()
####################################################
# PART 4: Two-view triangulation
####################################################
print "=====two-view triangulation======"
cam_intrinsic = np.matrix([[2759.48, 0, 1520.69],[0, 2764.16, 1006.81],[0, 0, 1]])

# compute essential matrix based on fundamental matrix
E1 = cv.findEssentialMat(points1, points2, cam_intrinsic, method=cv.RANSAC)[0]
print "essential matrix = "
print E1

# compute essential matrix in perspective of cam 2
E2 = cv.findEssentialMat(points2, points1, cam_intrinsic, method=cv.RANSAC)[0]
#print "essential matrix cam 2 = "
#print E2

# find rotation and transformation matrix based upon essential matrix
R1, R2, T = cv.decomposeEssentialMat(E1)
R_T_1 = np.hstack([R1,T])
print "[R|T] matrix = "
print R_T_1

R1, R2, T = cv.decomposeEssentialMat(E1)
R_T_2 = np.hstack([R1,T])
#print "cam2 [R|T] matrix = "
#print R_T_2

# s*[px_x, px_y,0] = cam_intrinsic*R_T*[X,Y,Z,0]
cam1_proj = np.matmul(E1,R_T_1)
cam2_proj = np.matmul(E2,R_T_2)
#print cam1_proj
#print cam2_proj

points4D = cv.triangulatePoints(cam1_proj,cam2_proj,points1[0],points2[0])
#print points4D

#print np.matmul(cam1_proj,points4D)

####################################################
# TASK 2-1 : Compute fundamental matrix for all image pairs
####################################################
print "=====fundamental matrix for all image pairs======"

#load all images
images = []
for i in range(0,9):
    images.append(cv.imread("images/000"+str(i)+".png"))
images.append(cv.imread("images/0010.png"))
print "10 images loaded"

#compute keypoints and descriptors for all images
sift = cv.xfeatures2d.SIFT_create(60)
keypoints = []
descriptors = []
i=0
for im in images:
    print "computing SIFT keypoints for image",i
    i+=1
    cv.resize(im, (0,0), fx=0.25, fy=0.25)
    k, d = sift.detectAndCompute(im,None)
    keypoints.append(k)
    descriptors.append(d)

# feature matching for all image pairs and compute fundamental matrix
F_matrices = []

for i in range(0,10):
    for j in range(i+1,10):
        nnMatches = findNNMatch(descriptors[i],descriptors[j],ratioTest=0.8)
        nnMatches = sorted(nnMatches, key = lambda x:x.distance)
        nnMatches = nnMatches[0:15]
        pts1 = []
        pts2 = []
        for m in nnMatches:
            pts1.append(keypoints[i][m.queryIdx].pt)
            pts2.append(keypoints[j][m.trainIdx].pt)
        points1 = np.int32(points1)
        points2 = np.int32(points2)
        F, _ = cv.findFundamentalMat(points1,points2,cv.FM_RANSAC)
        print "RANSAC fundamental matrix for image pair",i,",",j
        print F

        #F_matrices.append({i,j,F})
