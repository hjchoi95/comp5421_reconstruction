import math
import numpy as np
import cv2 as cv

# @output: calculates euclidean distance between two vectors in scalar
# @req: vec1 and vec2 must be in same length
def euclideanDistance(vec1, vec2):
    squared_total = 0
    for i in range(0,len(vec1)):
        squared_total += (vec1[i]-vec2[i])*(vec1[i]-vec2[i])
    return math.sqrt(squared_total)


def findNNMatch(des1, des2, ratioTest=0.8, threshold=10000):
    nnMatches = []

    for i in range(0,len(des1)):
        #print i
        currMin = threshold #threshold
        secondMin = currMin
        bestMatch = 0
        for j in range(0,len(des2)):
            dist = euclideanDistance(des1[i], des2[j])
            #print dist,bestMatch
            if (dist<currMin):
                secondMin = currMin
                currMin = dist
                bestMatch = j

        ##left-right check is not performed because it only pruned 6 entries out of 400 feature matches
        #ratio test alone does enough work, 199/400 selected
        if currMin/secondMin<0.8:
            nnMatches.append(cv.DMatch(i,bestMatch,0,currMin))
            #print "dist=",currMin,"2ndDist=",secondMin,"ratio=",currMin/secondMin,"imgIdx=",0,"queryIdx=",i,"trainIdx",bestMatch
    return nnMatches


# 8-point algorithm
def computeFundamentalMatrix(v1, v2, F):
    #form constraint matrix
    A = np.matrix([])#(np.zeros(shape=(9,9)))

    for i in range(0,len(v1)):
        #print "d"
        c_row = np.matrix([v2[i][0]*v1[i][0], v2[i][0]*v1[i][1], v2[i][0], v2[i][1]*v1[i][0], v2[i][1]*v1[i][0], v2[i][1], v1[i][0], v1[i][1], 1 ])
        if i==0:
            A = np.matrix(c_row)
        else:
            A = np.vstack([A,c_row])
    #print A

    #solve for SVD
    U,S,V = np.linalg.svd(A)
    V = V.conj().T
    F = V[:,8].reshape(3,3).copy()

    U,D,V = np.linalg.svd(F)
    F = np.dot(np.dot(U,np.diag([D[0], D[1], 0])),V)

    #print F
    return F


def drawlines(img1,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    #print "======= drawLines() =========="
    r, c, _ = img1.shape
    result = img1.copy()
    #img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    #img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        #print "drawline",x0,",",y0,"->",x1,",",y1
        result = cv.line(img1, (x0,y0), (x1,y1), color,2)
        result = cv.circle(img1,tuple(pt1),5,color,-1)
        #img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    #img1 = cv.line(img1, (0,0), (400,400),color,4)
    return result
