import keras
import tensorflow as tf
from keras.layers import Input,GRU,LSTM,Dense,Conv2D,AveragePooling1D,TimeDistributed,Flatten,MaxPooling2D,MaxPooling1D,Convolution1D,Reshape,Dropout,Embedding,Permute,Lambda,Multiply
from keras.layers import Bidirectional
from keras.models import Model 
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D
from keras.callbacks import EarlyStopping

from keras import backend as K
from keras.engine.topology import Layer
#===========================================================================================
#NSMN
from keras import backend as k
from keras.engine.topology import Layer

class nsmnattention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(nsmnattention, self).__init__(**kwargs)

    def build(self, input_shape):
        
       
        self.kernelW = self.add_weight(name='Wall', 
                                      shape=(10, 10),
                                      initializer='uniform',
                                      trainable=False)
        self.kernelWs = self.add_weight(name='Ws', 
                                      shape=(12,12),
                                      initializer='uniform',
                                      trainable=False)
        self.kernelWc = self.add_weight(name='Wc', 
                                      shape=(500,500),
                                      initializer='uniform',
                                      trainable=False)
        self.kernelas = self.add_weight(name='Was', 
                                      shape=(10,1),
                                      initializer='uniform',
                                      trainable=False)
        self.kernelac = self.add_weight(name='Wac', 
                                      shape=(10,1),
                                      initializer='uniform',
                                      trainable=False)
        super(nsmnattention, self).build(input_shape)  


    def call(self, x):
        U=Permute((2,1))(x[0])
        V=Permute((2,1))(x[1])
        print("U.shape",U.shape)
        print("V.shape",V.shape)
        
        E=k.batch_dot(Permute((2,1))(U),V)
        
        print("E.shape",E.shape)
        
        U1=k.batch_dot(V,Permute((2,1))((E)))     
        
        V1=k.batch_dot(U,E)

        U=Permute((2,1))(U)
        U1=Permute((2,1))(U1)
        V=Permute((2,1))(V)
        V1=Permute((2,1))(V1)
        S=Permute((2,1))((keras.layers.concatenate([U,U1,(U-U1),Multiply()([U,U1])])))
        T=Permute((2,1))((keras.layers.concatenate([V,V1,(V-V1),Multiply()([V,V1])])))
        print("S.shape",S.shape)
        print("T.shape",T.shape)
                        
        P=LSTM(10,return_sequences=True)(S)
        Q=LSTM(10,return_sequences=True)(T)
        print("P.shape",P.shape)
        print("Q.shape",Q.shape)
                  
        p=MaxPooling1D((40))(P)
        q=MaxPooling1D((40))(Q)
        
        print("p.shape",p.shape)
        print("q.shape",q.shape)
        
        m=keras.layers.concatenate([p,q,(p-q),Multiply()([p,q])])
        print("m.shape",m.shape)
        
        return m

    def compute_output_shape(self, input_shape):
        return (None, 40)
#============================================================================================
#co-attention
from keras import backend as k
from keras.engine.topology import Layer

class coattention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(coattention, self).__init__(**kwargs)

    def build(self, input_shape):
        
        
        self.kernelW = self.add_weight(name='Wall', 
                                      shape=(10, 10),
                                      initializer='uniform',
                                      trainable=True)
        self.kernelWs = self.add_weight(name='Ws', 
                                      shape=(12,12),
                                      initializer='uniform',
                                      trainable=True)
        self.kernelWc = self.add_weight(name='Wc', 
                                      shape=(100,100),
                                      initializer='uniform',
                                      trainable=True)
        self.kernelas = self.add_weight(name='Was', 
                                      shape=(10,1),
                                      initializer='uniform',
                                      trainable=True)
        self.kernelac = self.add_weight(name='Wac', 
                                      shape=(10,1),
                                      initializer='uniform',
                                      trainable=True)
        super(coattention, self).build(input_shape)  # 一定要在最后调用它


    def call(self, x):
        C=x[0]
       
        print("C.shape",C.shape)
        RNN=Permute((2,1))(x[1])
        
        f=k.dot(C,self.kernelW)
        print("f.shape",f.shape)
        F=k.tanh(k.batch_dot(f,RNN))
        print("F.shape",F.shape)
        
        s=k.dot(RNN,self.kernelWs)
        print("s.shape",s.shape)
        c=k.dot(Permute((2,1))(C),self.kernelWc)
        print("c.shape",c.shape)
       
        Hs=k.tanh(s+k.batch_dot(c,F))
        print("Hs.shape",Hs.shape)
        Hc=k.tanh(c+k.batch_dot(s,Permute((2,1))(F)))
        print("Hc.shape",Hc.shape)
        
        
        As=k.softmax(k.dot(Permute((2,1))(Hs),self.kernelas))
        print("As.shape",As.shape)
        Ac=k.softmax(k.dot(Permute((2,1))(Hc),self.kernelac))
        print("Ac.shape",Ac.shape)
        
        As=Permute((2,1))(As)
        print("As.shape",As.shape)
        Ac=Permute((2,1))(Ac)
        print("Ac.shape",Ac.shape)
        
        sfinal=k.batch_dot(As,Permute((2,1))(RNN))
        
        print("sfinal.shape",sfinal.shape)
        cfinal=k.batch_dot(Ac,C)
       
        print("cfinal.shape",cfinal.shape)
        
        return keras.layers.concatenate([sfinal,cfinal])

    def compute_output_shape(self, input_shape):
        return (None, 20)

#=================================================================================
#train the model
import numpy as np
from keras import regularizers
y=f["label"].tolist()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(plain_e,y,test_size=0.2,random_state=1000)
X_train_1,X_test_1,y_train,y_test=train_test_split(wikidata_e,y,test_size=0.2,random_state=1000)

from keras.utils.np_utils import to_categorical
y_train= to_categorical(y_train,2)
y_train= y_train.astype('int')
y_test= to_categorical(y_test,2)
y_test= y_test.astype('int')
import numpy as np


winput=Input(shape=(100,50))
wembed=LSTM(10,return_sequences=True)(winput)

winput_1=Input(shape=(12,50))
wembed_1=LSTM(10,return_sequences=True)(winput_1)

co=nsmnattention(40)([wembed,wembed_1])
co=Dense(2)(co)
coc=coattention(20)([wembed,wembed_1])
coc=Dense(2)(coc)
c=keras.layers.concatenate([co,coc])
output=Dense(2)(c)
output=Dense(2,activation="softmax")(output)

model = Model([winput,winput_1], [output])
model.summary()

RMSprop=keras.optimizers.Adam(lr=0.01)
model.compile(optimizer=RMSprop,loss="categorical_crossentropy",metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
history=model.fit([np.array(X_train),np.array(X_train_1)],[np.array(y_train)],
                  epochs=20,validation_split=0.2, callbacks=[early_stopping], batch_size=32)
scores=model.evaluate([np.array(X_test),np.array(X_test_1)],np.array(y_test), verbose=0)
pre=model.predict([np.array(X_test),np.array(X_test_1)])
print(scores)
#======================================================================================
#evaluation metrics

from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score
from sklearn.metrics import confusion_matrix
y_pre=[]
for i in range(len(pre)):
    k=pre[i]
    w=np.where(k==np.max(k))[0][0].tolist()
    y_pre.append(w)

    
y=f["label"].tolist()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(plain_e,y,test_size=0.2,random_state=1000)
X_train_1,X_test_1,y_train,y_test=train_test_split(wikidata_e,y,test_size=0.2,random_state=1000)

print(confusion_matrix(y_test, y_pre))

print('Weighted precision', precision_score(y_test, y_pre,labels=[1], average='macro'))
print('Weighted recall', recall_score(y_test, y_pre, labels=[1], average='macro'))
print('Weighted f1-score', f1_score(y_test, y_pre, labels=[1], average='macro'))

#calculate the precision@50
a=set(np.argsort(np.array(y_pre)).tolist()[len(y_test)-50:len(y_test)])
a=list(a)
p=[]
for i in range(50):
    g=a[i]
    p.append(y_test[g])
pre50=np.sum(p)/50

print(pre50)
