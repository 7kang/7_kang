#coding=utf8
import itchat, time
from itchat.content import *
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

@itchat.msg_register([PICTURE])
def download_files(msg):
    msg['Text']('mimi.jpg')
    img_path = 'mimi.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    ans=decode_predictions(preds, top=2)[0][0][1]
    print('Predicted:', ans)
    itchat.send_msg(ans,msg['FromUserName'])

itchat.auto_login()
itchat.run()


