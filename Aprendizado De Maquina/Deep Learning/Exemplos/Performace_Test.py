'''Test your model performance on custom data:'''
'''Define image properties:'''
Image_Width=128
Image_Height=128
Image_Size=(Image_Width,Image_Height)
Image_Channels=3

results={
    0:'cat',
    1:'dog'
}
from PIL import Image
import numpy as np
im=Image.open("dogs-vs-cats\Custom\pic01.jpg")
im=Image.open("__image_path_TO_custom_image")
im=im.resize(Image_Size)
im=np.expand_dims(im,axis=0)
im=np.array(im)
im=im/255
# pred=model.predict_classes([im])[0]
pred=model.predict([im])[0]
print(pred,results[pred])
