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


def Package():
    Save_path = 'Input/'
    image_list = glob.glob('CoarseData/CoarseData/*/*.png')
    data = np.zeros((len(image_list),224,224,3),dtype=np.uint8)
    label = np.zeros((len(image_list),185))
    for i in range(len(image_list)):
        print(i)
        path_ = image_list[i]
        img = cv2.imread(path_)

        f = open(path_[:len(path_)-8]+'.txt')
        content = f.readlines()
        content= [x.strip() for x in content]
        label[i,:100] = np.array(content[0].split(" "),dtype = np.float)
        label[i,100:179] = np.array(content[1].split(" "),dtype = np.float)
        label[i,179:] = np.array(content[2].split(" "),dtype = np.float)
        f.close()
        data[i,:,:] = np.array(img,dtype=np.uint8)

    np.save(Save_path+'data.npy',data)
    np.save(Save_path+'label.npy',label)

def Split():
    test_num = 10000
    data = np.load('Input/data.npy')
    label= np.load('Input/label.npy')
    train_data= data[test_num:,:]
    test_data = data[:test_num,:]
    test_label = label[:test_num,:]
    train_label = label[test_num:,:]
    np.save('Input/train_data.npy',train_data)
    np.save('Input/mean_data.npy',np.mean(data,axis=0))
    np.save('Input/mean_label.npy', np.mean(label, axis=0))
    np.save('Input/test_data.npy',test_data)
    np.save('Input/train_label.npy',train_label)
    np.save('Input/test_label.npy',test_label)

#Split()
