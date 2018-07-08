import numpy as np
import dlib
import glob
import cv2
import os

os.chdir('/media/songguoxian/DATA/UnixFolder/3DMM/Coarse_Dataset/Coarse_Dataset/')

def LoadBase():
    predictor_path = '/home/songguoxian/Desktop/Library/dlib-19.4/python_examples/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    return predictor,detector

def SaveLandmark(shape):
    tmp = np.zeros((68,2),dtype=np.uint)
    for i in range(68):
        tmp[i,0] = shape.part(i).x
        tmp[i, 1] = shape.part(i).y
    return tmp



def Run():
    count =0
    predictor, detector  = LoadBase()
    image_list = glob.glob('CoarseData/CoarseData/*/*.jpg')
    for path_ in image_list:
        print(count)
        count+=1
        image_= cv2.imread(path_)
        gray_= cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)
        try:
            dets = detector(gray_, 1)
            shape = predictor(gray_, dets[0])
            tmp = SaveLandmark(shape)
            np.savetxt(path_[:len(path_)-4]+'_landmark.txt',tmp, fmt='%d')
            res = Normalize(image_,tmp)
            cv2.imwrite(path_[:len(path_)-4]+'_224.png',res)
        except:
            continue

def Normalize(image,landmark_):
    xmin = np.min(landmark_[:,0])
    xmax = np.max(landmark_[:,0])
    ymin = np.min(landmark_[:,1])
    ymax = np.max(landmark_[:,1])
    sub_image = image[ymin:ymax,xmin:xmax]
    res = cv2.resize(sub_image, (224,224), interpolation=cv2.INTER_LINEAR)
    return res


Run()