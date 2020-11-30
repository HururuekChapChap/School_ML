#!/usr/bin/env python
# coding: utf-8

# In[21]:


#import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K


# In[1]:


class LeNet:
    
    def build(width,height,depth,classes):
        
        #initialize the model
        model = Sequential()
        inputShape = (height,width,depth)
        
        #if we are using "channels first", update teh input shape
#         if K.image_data_format() == "channels_first":
#             inputShape = (depth,height,width)
            
        #first set of CONV => RELU => POOL layers
        model.add(Conv2D(20,(5,5), padding="same", input_shape=inputShape))
        model.add(Activation("relu")) #원래는 하이퍼 탄젠트인데 우리는 렐루를 쓴다
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        
        #secode set of CONV => RELU => POOL layers
        model.add(Conv2D(50,(5,5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        
        #first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        
        #sofmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        #return the constructed network architectrue
        return model
        


# In[ ]:





# In[ ]:




