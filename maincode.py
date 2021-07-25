import tensorflow as tf
import pandas as pd
import numpy as np




#서버에서 받아오는 영역
gender = input('성별을 입력해주세요: ')
age = int(input('나이를 입력해주세요: '))
season = int(input('달을 입력해주세요 :'))

if season >= 3 :
    if season <= 5 :
        season = 13.2
        
elif season >= 6 :
    if season <= 8 :
        season = 24.8
        
elif season >= 9 :
    if season <= 11 :
        season = 15.1
        
elif season >= 12 :
    if season <= 2 :
            season =1.6
    else :
                print("1-12까지의 숫자를 입력해주세요")

b2 = season





if age <= 5 :
    age = 2.5
elif age <= 10 :
    age = 7.5
    
elif age <= 15 :
    age = 12.5
    
elif age <= 20 :
    age = 17.5
    
elif age <= 25 :
    age = 22.5
    
elif age <= 30 :
    age = 27.5
    
elif age <= 35 :
    age = 32.5

elif age <= 40 :
    age = 37.5

elif age <= 45 :
    age = 42.5

elif age <= 50 :
    age = 47.5

elif age <= 55 :
    age = 52.5

elif age <= 60 :
    age = 57.5

elif age <= 65 :
    age = 62.5

elif age <= 70 :
    age = 67.5

elif age <= 75 :
    age = 72.5

elif age <= 80 :
    age = 77.5

else :
    if age >= 80 :
        age = 80
    else :
            if age <= 100 :
                print("100세 이상은 측정이 불가능합니다")

b1 = age


                
    
    

if gender == "남" :
    gender = 0
elif gender == "여" :
    gender = 1
else :
    printf("남과 여로 입력해주세요")

b = gender

global a
global name
c = []
name_index = []


#병의 개수로 n개를 정해서 n번의 루프문만 돌리는 것이 최적
#데이터의 개수 구하는 명령문 이용해서 가능 먼저 10개만 만듬


def hepatitis(b,b1,b2):

    name = "만성간염"
    
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


    model.fit(독립, 종속, epochs=1000, verbose=0)
    model.fit(독립, 종속, epochs=1000)
    model.fit(독립, 종속, epochs=1000)



    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open('hepatitismodel.tflite', 'wb') as f:
      f.write(tflite_model)



    print(model.predict([[b,b1,b2]]))

    a = model.predict([[b,b1,b2]])

    return a

    print(a)

    #print(종속[0:5])
 
    #print(model.get_weights())


def Osteomyelitis(b,b1,b2):

    name = "골수염"
    
    병확률 = pd.read_csv('골수염.txt',sep = "\t")
    print(병확률.columns)
    병확률.head()


    독립 = 병확률[['성별' , '나이', '계절(기온)']]
    종속 = 병확률[['확률']]
    print(독립.shape, 종속.shape)


    X = tf.keras.layers.Input(shape=[3])
    Y = tf.keras.layers.Dense(1)(X)
    model = tf.keras.models.Model(X, Y)
    model.compile(optimizer="Adam", loss="mse")


    model.fit(독립, 종속, epochs=1000, verbose=0)
    model.fit(독립, 종속, epochs=1000)
    model.fit(독립, 종속, epochs=1000)


    a = model.predict([[b,b1,b2]])

    print(a)

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open('Osteomyelitismodel.tflite', 'wb') as f:
      f.write(tflite_model)

    return a

    #print(종속[0:5])
 


def High_blood_pressuremodel(b,b1,b2):
    
    병확률 = pd.read_csv('고혈압.txt',sep = "\t")
    print(병확률.columns)
    병확률.head()


    독립 = 병확률[['성별' , '나이', '계절(기온)']]
    종속 = 병확률[['확률']]
    print(독립.shape, 종속.shape)


    X = tf.keras.layers.Input(shape=[3])
    Y = tf.keras.layers.Dense(1)(X)
    model = tf.keras.models.Model(X, Y)
    model.compile(optimizer="Adam", loss="mse")


    model.fit(독립, 종속, epochs=1000, verbose=0)
    model.fit(독립, 종속, epochs=1000)
    model.fit(독립, 종속, epochs=1000)


    a = model.predict([[b,b1,b2]])

    print(a)

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open('High_blood_pressuremodel_model.tflite', 'wb') as f:
      f.write(tflite_model)
                                 

    return a

    #print(종속[0:5])
 


def Vitiligo(b,b1,b2):
    
    병확률 = pd.read_csv('백반증.txt',sep = "\t")
    print(병확률.columns)
    병확률.head()


    독립 = 병확률[['성별' , '나이', '계절(기온)']]
    종속 = 병확률[['확률']]
    print(독립.shape, 종속.shape)


    X = tf.keras.layers.Input(shape=[3])
    Y = tf.keras.layers.Dense(1)(X)
    model = tf.keras.models.Model(X, Y)
    model.compile(optimizer="Adam", loss="mse")


    model.fit(독립, 종속, epochs=1000, verbose=0)
    model.fit(독립, 종속, epochs=1000)
    model.fit(독립, 종속, epochs=1000)


    a = model.predict([[b,b1,b2]])

    print(a)

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open('Vitiligomodel.tflite', 'wb') as f:
      f.write(tflite_model)
    

    return a
    
    #print(종속[0:5])
 


def polio(b,b1,b2):
    
    병확률 = pd.read_csv('소아마비.txt',sep = "\t")
    print(병확률.columns)
    병확률.head()


    독립 = 병확률[['성별' , '나이', '계절(기온)']]
    종속 = 병확률[['확률']]
    print(독립.shape, 종속.shape)


    X = tf.keras.layers.Input(shape=[3])
    Y = tf.keras.layers.Dense(1)(X)
    model = tf.keras.models.Model(X, Y)
    model.compile(optimizer="Adam", loss="mse")


    model.fit(독립, 종속, epochs=1000, verbose=0)
    model.fit(독립, 종속, epochs=1000)
    model.fit(독립, 종속, epochs=1000)


    a = model.predict([[b,b1,b2]])

    print(a)

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open('polio.tflite', 'wb') as f:
      f.write(tflite_model)

    return a
    
    #print(종속[0:5])
 


def Hydrocephalus(b,b1,b2):
    
    병확률 = pd.read_csv('수두증.txt',sep = "\t")
    print(병확률.columns)
    병확률.head()


    독립 = 병확률[['성별' , '나이', '계절(기온)']]
    종속 = 병확률[['확률']]
    print(독립.shape, 종속.shape)


    X = tf.keras.layers.Input(shape=[3])
    Y = tf.keras.layers.Dense(1)(X)
    model = tf.keras.models.Model(X, Y)
    model.compile(optimizer="Adam", loss="mse")


    model.fit(독립, 종속, epochs=1000, verbose=0)
    model.fit(독립, 종속, epochs=1000)
    model.fit(독립, 종속, epochs=1000)


    a = model.predict([[b,b1,b2]])

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open('Hydrocephalus.tflite', 'wb') as f:
      f.write(tflite_model)


    print(a)

    return a

    #print(종속[0:5])
 


def Alzheimer(b,b1,b2):
    
    병확률 = pd.read_csv('알츠하이머.txt',sep = "\t")
    print(병확률.columns)
    병확률.head()


    독립 = 병확률[['성별' , '나이', '계절(기온)']]
    종속 = 병확률[['확률']]
    print(독립.shape, 종속.shape)


    X = tf.keras.layers.Input(shape=[3])
    Y = tf.keras.layers.Dense(1)(X)
    model = tf.keras.models.Model(X, Y)
    model.compile(optimizer="Adam", loss="mse")


    model.fit(독립, 종속, epochs=1000, verbose=0)
    model.fit(독립, 종속, epochs=1000)
    model.fit(독립, 종속, epochs=1000)


    a = model.predict([[b,b1,b2]])

    print(a)

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open('Alzheimer.tflite', 'wb') as f:
      f.write(tflite_model)

    return a

    #print(종속[0:5])
 



def Hypothermia(b,b1,b2):
    
    병확률 = pd.read_csv('저체온증.txt',sep = "\t")
    print(병확률.columns)
    병확률.head()


    독립 = 병확률[['성별' , '나이', '계절(기온)']]
    종속 = 병확률[['확률']]
    print(독립.shape, 종속.shape)


    X = tf.keras.layers.Input(shape=[3])
    Y = tf.keras.layers.Dense(1)(X)
    model = tf.keras.models.Model(X, Y)
    model.compile(optimizer="Adam", loss="mse")


    model.fit(독립, 종속, epochs=1000, verbose=0)
    model.fit(독립, 종속, epochs=1000)
    model.fit(독립, 종속, epochs=1000)


    a = model.predict([[b,b1,b2]])

    print(a)

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open('Hypothermia.tflite', 'wb') as f:
      f.write(tflite_model)

    return a

    #print(종속[0:5])
 


def dementia(b,b1,b2):
    
    병확률 = pd.read_csv('치매.txt',sep = "\t")
    print(병확률.columns)
    병확률.head()


    독립 = 병확률[['성별' , '나이', '계절(기온)']]
    종속 = 병확률[['확률']]
    print(독립.shape, 종속.shape)


    X = tf.keras.layers.Input(shape=[3])
    Y = tf.keras.layers.Dense(1)(X)
    model = tf.keras.models.Model(X, Y)
    model.compile(optimizer="Adam", loss="mse")


    model.fit(독립, 종속, epochs=1000, verbose=0)
    model.fit(독립, 종속, epochs=1000)
    model.fit(독립, 종속, epochs=1000)


    a = model.predict([[b,b1,b2]])

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open('dementia.tflite', 'wb') as f:
      f.write(tflite_model)

    print(a)

    return a

    #print(종속[0:5])
 



def Alopecia(b,b1,b2):
    
    병확률 = pd.read_csv('탈모증.txt',sep = "\t")
    print(병확률.columns)
    병확률.head()


    독립 = 병확률[['성별' , '나이', '계절(기온)']]
    종속 = 병확률[['확률']]
    print(독립.shape, 종속.shape)


    X = tf.keras.layers.Input(shape=[3])
    Y = tf.keras.layers.Dense(1)(X)
    model = tf.keras.models.Model(X, Y)
    model.compile(optimizer="Adam", loss="mse")


    model.fit(독립, 종속, epochs=1000, verbose=0)
    model.fit(독립, 종속, epochs=1000)
    model.fit(독립, 종속, epochs=1000)


    a = model.predict([[b,b1,b2]])

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open('Alopecia.tflite', 'wb') as f:
      f.write(tflite_model)


    print(a)

    return a

    #print(종속[0:5])
 



def Menopause(b,b1,b2):
    
    병확률 = pd.read_csv('폐경.txt',sep = "\t")
    print(병확률.columns)
    병확률.head()


    독립 = 병확률[['성별' , '나이', '계절(기온)']]
    종속 = 병확률[['확률']]
    print(독립.shape, 종속.shape)


    X = tf.keras.layers.Input(shape=[3])
    Y = tf.keras.layers.Dense(1)(X)
    model = tf.keras.models.Model(X, Y)
    model.compile(optimizer="Adam", loss="mse")


    model.fit(독립, 종속, epochs=1000, verbose=0)
    model.fit(독립, 종속, epochs=1000)
    model.fit(독립, 종속, epochs=1000)



    if b == 0 :
        a = 0
    else :
        a = model.predict([[b,b1,b2]])


    print(a)

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open('Menopause.tflite', 'wb') as f:
      f.write(tflite_model)


    #print(종속[0:5])

    return a
 




for i in range(0, 10, 1):
    
                     
    name = input('병코드을 입력해주세요: ')


    if name == '만성간염':
        a = hepatitis(b,b1,b2)

    elif name == '고혈압':
        a = Osteomyelitis(b,b1,b2)
    
    elif name == '골수염':
        a = High_blood_pressuremodel(b,b1,b2)
    
    elif name == '백반증':
        a = Vitiligo(b,b1,b2)
    
    elif name == '소아마비':
        a = polio(b,b1,b2)
    
    elif name == '알츠하이머':
        a = Hydrocephalus(b,b1,b2)
    
    elif name == '수두증':
        a = Alzheimer(b,b1,b2)
    
    elif name == '저체온증':
        a = Hypothermia(b,b1,b2)

    elif name == '치매':
        a = dementia(b,b1,b2)

    elif name == '탈모증':
        a = Alopecia(b,b1,b2)

    elif name == '폐경':
        a = Menopause(b,b1,b2)

    else :
        print("병정보 없음")


    c.insert(i,a)


    name_index.insert(i,name)


    i = i+1

    if i == 10 :

        y = c[0]
        y1 = c[1]
        y2 = c[2]
        y3 = c[3]
        y4 = c[4]
        y5 = c[5]
        y6 = c[6]
        y7 = c[7]
        y8 = c[8]
        y9 = c[9]

        x = name_index[0]
        x1 = name_index[1]
        x2 = name_index[2]
        x3 = name_index[3]
        x4 = name_index[4]
        x5 = name_index[5]
        x6 = name_index[6]
        x7 = name_index[7]
        x8 = name_index[8]
        x9 = name_index[9]

        result = [[x,y],[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6],[x7,y7],[x8,y8],[x9,y9]]

        result.sort(key=lambda x:-x[1])
        print(result)



#2차원 요소 a[세로인덱스][가로인덱스] <-   0,0 = x    0,1 = y      result[0]=[x,y]
#answer.insert(0, 1) (answer.append(i, 입력한 숫자)로 일반화) 내지는 answer.append(1) (answer.append(입력한 숫자)로 일반화)































    
