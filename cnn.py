#  CNN Model

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28, 28, 1), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten())

# 128 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(128, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# You can play around with number of epochs
# YOUR ACCURACY MAY ALSO BE HIGHER THAN WHAT IS SHOWN HERE 
model.fit(x_train,y_train,epochs=10)

# 5.1.2 Displaying mode architecture
model.summary()

# Evaluation of CNN model

model.metrics_names
model.evaluate(x_test,y_test)

#Saving the model
model.save('mnist_cnn.h5')


