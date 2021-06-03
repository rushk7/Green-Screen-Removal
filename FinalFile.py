from scipy import misc,ndimage
from sklearn.externals import joblib
from PIL import Image
import numpy as np
from PIL import Image
import numpy as np
import cv2
from scipy import misc


def green_screen(model,image,bg_image) :
    l = image.shape[0]
    h = image.shape[1]
    bg_image = bg_image[:l,:h]
    print(bg_image.shape)
    print(image.shape)
    image = image.reshape((l*h,3))
    
    bg_image = bg_image.reshape((l*h,3))
    #     z = np.ones((l*h,4))
    #     z[:,:-1] = image
    result = model.predict(image)
    for i,x in enumerate(result) :
        if x == 1:
            image[i] = bg_image[i]


    image = image.reshape((l,h,3))
#     z = z.reshape((l,h,4))
    return image


def return_model(model_address) :
    model = joblib.load(model_address)

    return model

def getImage(image_add) :
    image = misc.imread(image_add)
    return image

def setImage(image,image_add) :
    misc.imsave(image_add,image)


# input :- address of the old and new video
def make_video(video_add,res_video_add) :
    vidcap = cv2.VideoCapture(video_add)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video = cv2.VideoWriter(res_video_add,cv2.VideoWriter_fourcc(*'mp4v') , fps, (width,height),True)


    success,image = vidcap.read()
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        if image == None :
            break
        image = green_screen(model,image,bg_image)
        video.write(image)
        count += 1
        print(count)
    video.release()



def convert_image_to_data(image1,image2) :
    y = []
    pix_val1 = list(image2.getdata())
    pix_val = list(image1.getdata())
    data = np.asarray( image1, dtype="int32" )
    data1 = np.asarray( image2, dtype="int32" )
    data = data.reshape((data.shape[0]*data.shape[1],3))
    data1 = data1.reshape((data1.shape[0]*data1.shape[1],3))
    #     for i,x in enumerate(data1) :
    #         if x[0] > 240 and x[1] > 240 and x[2] > 240 :
    #             y.append(0)
    #         else :
    #             y.append(1)
    
    for i in range(data.shape[0]) :
        if data1[i][0] > 240 and data1[i][1] > 240 and data1[i][2] > 240 :
            y.append(0)
        else :
            y.append(1)


    return data,y






