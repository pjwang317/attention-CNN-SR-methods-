import numpy as np
from scipy import ndimage
import scipy.misc
import skimage.color

def avegrad(img):
    img = np.double(img)
    H, W, C = img.shape
    print (C)
    ave_grad = np.zeros([1,C], dtype = np.float32)

    for i in range(C):
        ### save the gradient value of single pixel
        gradval = np.zeros([H, W],dtype = np.float32)
        ###save  the differential value of X orient
        diffX = np.zeros([H, W],dtype = np.float32)
        ### save the differential value of Y orient
        diffY = np.zeros([H, W],dtype = np.float32)
        tempX = np.zeros([H, W],dtype = np.float32)
        tempY = np.zeros([H, W],dtype = np.float32)
        tempX[0:H-1,0:W-2] = img[0:H-1,1:W-1,i]
        tempY[0:H-2,0:W-1] =img[1:H-1,0:W-1,i]
        diffX = tempX - img[:,:,i]
        diffY = tempY - img[:,:,i]
        diffX[0:H-1,W-1] = 0
        diffY[H-1,1:W-1] = 0
        diffX = diffX*diffX
        diffY = diffY*diffY
        avegrad = (diffX + diffY)/2
        avegrad = np.sum(np.sum(np.sqrt(avegrad)))
        avegrad = avegrad/((H-1)*(W-1))

        ave_grad[0,i] = avegrad

    return np.mean(ave_grad)
        


# ref = scipy.misc.imread('./test_imgs/rgb_001.png')
# dis = scipy.misc.imread('./test_imgs/bic_rgb_001.png')
# ref_ag = avegrad(ref)
# dis_ag = avegrad(dis)
# print (ref_ag)
# print (dis_ag)


