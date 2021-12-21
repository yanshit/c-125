import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from PIL import Image
import PIL.ImageOps


X,y=fetch_openml('mnist_784',version=1,return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=9,train_size=7500,test_size=2500)
X_train_scale=X_train/255.0
X_test_scale=X_test/255.0


clf=LogisticRegression(solver='saga',   multi_class='multinomial').fit(X_train_scale,y_train)

def getPrediction(image):
    im_pil=Image.open(image)
    image_bw=im_pil.convert('L')
    image_bw_resize=image_bw.resize((28,28),Image.ANTIALIAS)
    pixelfilter=20
    min_pix=np.percentile(image_bw_resize,pixelfilter)
    image_bw_resize_inverted=np.clip(image_bw_resize-min_pix,0,255)
    max_pix=np.max(image_bw_resize)
    image_bw_resize_inverted=np.asarray(image_bw_resize_inverted)/max_pix
    test_sample=np.array(image_bw_resize_inverted).reshape(1,784)
    test_pred=clf.predict(test_sample)
    return test_pred[0]
