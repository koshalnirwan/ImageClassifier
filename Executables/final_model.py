#import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, Dense, Flatten
from keras.callbacks import TensorBoard
import pickle
import datetime
import time
 
X = pickle.load(open('X3.pickle','rb'))
y = pickle.load(open('y3.pickle','rb'))

X = X/255.0

dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size,dense_layer, int(time.time())) 
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
            
            model = Sequential()
            
            model.add(Conv2D(layer_size,(3,3),activation = 'relu',input_shape = X.shape[1:]))
            model.add(MaxPooling2D(pool_size = (2,2)))
            
            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3,3), activation = 'relu'))
                model.add(MaxPooling2D(pool_size = (2,2)))
            
            model.add(Flatten())
            
            for l in range(dense_layer):
                model.add(Dense(layer_size, activation = 'relu'))
            
            model.add(Dense(1, activation = 'sigmoid'))
            
            model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
            
            model.fit(X,y, batch_size = 32,epochs = 10, validation_split = 0.2, callbacks = [tensorboard])
  

pickle_out = open('final_model.pickle','wb')
pickle.dump(model,pickle_out)
pickle_out.close()

model.save('64x3-myCNN.model')







