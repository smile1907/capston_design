import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def 만성간염() :
    
    병확률 = pd.read_csv('만성간염.txt',sep = "\t")
    print(병확률.columns)
    병확률.head()
 

    독립 = 병확률[['성별' , '나이', '계절(기온)']]
    종속 = 병확률[['확률']]
    print(독립.shape, 종속.shape)

 
    X = tf.keras.layers.Input(shape=[3])
    Y = tf.keras.layers.Dense(1)(X)
    model = tf.keras.models.Model(X, Y)
    model.compile(optimizer="Adam", loss="mse")


    history = model.fit(독립, 종속, epochs=1000, verbose=0)
    model.fit(독립, 종속, epochs=100)
 

    print(model.predict(독립[0:5]))

    print(종속[0:5])
 
    print(model.get_weights())

    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    #plot_model(model, to_file='model.png')
    #plot_model(model, to_file='model_shapes.png', show_shapes=True)



    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open('만성간염model.tflite', 'wb') as f:
      f.write(tflite_model)






