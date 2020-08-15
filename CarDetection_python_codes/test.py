
import sklearn
# Import the required modules
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
import argparse as ap
from nms import nms
from config import *

import numpy
import numpy as np
from sys import maxsize
from numpy import set_printoptions

set_printoptions(threshold=maxsize)

file1 = open('hog_features_test_100.txt','w')
# file2 = open('tried_impl_hog.txt','w')

def print_like_array(im_window):

    print('{',end='')
    for i in range(im_window.shape[0]):
        # print('{',end='')
        for j in range(im_window.shape[1]):
            print(im_window[i][j],end='')

            if j!=im_window.shape[1]-1:
                print(',',end='')
            # else:
                # print('}',end='')
        if i!=im_window.shape[0]-1:
            print(',',end='')
        # else:
    print('}',end='')

def write_to_file(fd,coefficients,intercept):
    

    for i in range(fd.shape[0]):
        file1.write(str(fd[i]))
        file1.write('\n')
    
    file1.write('coefs = \n')
    for i in range(coefficients.shape[0]):
        file1.write('{')
        for j in range(coefficients.shape[1]):
            file1.write(str(coefficients[i][j]))
            if j!= coefficients.shape[1]-1:
                file1.write(',')
            else:
                file1.write('}')
        if i!=coefficients.shape[0]-1:
            file1.write(',')
        # else:
        #     file1.write(']')
    
    file1.write('\nintercept = '+str(intercept[0]))



def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0) 
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    '''
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def main_func():
    # Parse the command line arguments
    # parser = ap.ArgumentParser()
    # parser.add_argument('-i', "--image", help="Path to the test image", required=True)
    # parser.add_argument('-d','--downscale', help="Downscale ratio", default=1.25,
    #         type=int)
    # parser.add_argument('-v', '--visualize', help="Visualize the sliding window",
    #         action="store_true")
    # args = vars(parser.parse_args())

    # Read the image
    args={}
    args["image"]='../CarData/CarData/TestImages/test-100.pgm'
    args['downscale']=1.25
    args['visualize']=False
    model_path = 'svm.model'

    im = imread(args["image"], as_gray=False)
    # print('im: ',im)
    np.savetxt('image_array.txt',im.flatten(),delimiter=',',newline=',')
    # min_wdw_sz = (100, 40)
    # step_size = (10, 10)
    downscale = args['downscale']
    visualize_det = args['visualize']

    # Load the classifier
    clf = joblib.load(model_path)

    # List to store the detections
    detections = []
    # The current scale of the image
    scale = 0

    cnt=0
    # Downscale the image and iterate
    for im_scaled in pyramid_gaussian(im, downscale=downscale):
        # This list contains detections at the current scale
        cd = []
        print('im.shape',im.shape)
        # If the width or height of the scaled image is less than
        # the width or height of the window, then end the iterations.
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue
            # Calculate the HOG features
            # print(im_window)
            fd =    hog(im_window, orientations, pixels_per_cell, cells_per_block, visualize=visualize, transform_sqrt=transform_sqrt,cnt=cnt)

            if cnt==0:
                # print('Hog features type: ',fd.shape)

                print('Hog features shape: ',fd.shape[0])

                write_to_file(fd,clf.coef_,clf.intercept_)
                # print(im_window.shape)
                print_like_array(im_window)

                cnt+=1
            # print('intercept = ',clf.intercept_)
            # fd2 = hog_features(im_window, pixels_per_cell, cells_per_block, signed_orientation=False, nbins=orientations, visualise=visualize, flatten=True, same_size=False)
            
            # print(fd.shape,fd2.shape)
            

            # file1.write("\n".join(" ".join(map(str, x)) for x in fd))
            # file2.write("\n".join(" ".join(map(str, x)) for x in fd2))

            # print('hog features: ',fd.dtype)
            fd = fd.reshape(1,-1)
            pred = clf.predict(fd)
            if pred == 1:
                print ("Detection:: Location -> ({}, {})".format(x, y))
                print('Decision function will now be called')
                print("Scale ->  {} | Confidence Score {} \n".format(scale,clf.decision_function(fd)))
                detections.append((x, y, clf.decision_function(fd),
                    int(min_wdw_sz[0]*(downscale**scale)),
                    int(min_wdw_sz[1]*(downscale**scale))))
                cd.append(detections[-1])
            # If visualize is set to true, display the working
            # of the sliding window
            if visualize_det:
                clone = im_scaled.copy()
                for x1, y1, _, _, _  in cd:
                    # Draw the detections at this scale
                    cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
                        im_window.shape[0]), (0, 0, 0), thickness=2)
                cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y +
                    im_window.shape[0]), (255, 255, 255), thickness=2)
                cv2.imwrite("Sliding_Window_in_Progress.pgm", clone)
                
        # Move the the next scale
        scale+=1

    # Display the results before performing NMS
    clone = im.copy()
    for (x_tl, y_tl, _, w, h) in detections:
        # Draw the detections
        cv2.rectangle(im, (x_tl, y_tl), (x_tl+w, y_tl+h), (0, 0, 0), thickness=2)
    cv2.imwrite("Raw_Detections_before_NMS.pgm", im)
    # cv2.waitKey()

    # Perform Non Maxima Suppression
    detections = nms(detections, threshold)

    # Display the results after performing NMS
    for (x_tl, y_tl, _, w, h) in detections:
        # Draw the detections
        cv2.rectangle(clone, (x_tl, y_tl), (x_tl+w,y_tl+h), (0, 0, 0), thickness=2)
    cv2.imwrite("Final_Detections_after_applying_NMS.pgm", clone)
    # cv2.waitKey()

main_func()