import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding,LSTM,Dense,Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


#poetry="No man is an island, Entire of itself, Every man is a piece of the continent, A part of the main. \n If a clod be washed away by the sea, Europe is the less. \n As well as if a promontory were. \n As well as if a manor of thy friend’s Or of thine own were: Any man’s death diminishes me, Because I am involved in mankind, And therefore never send to know for whom the bell tolls; It tolls for thee."

poetry=open("poem.txt").read()
corpus= poetry.lower().split("\n")
#vocab_size=30
#oov_tok="<OOV>"
tokenizer=Tokenizer()

tokenizer.fit_on_texts(corpus)
word_index=tokenizer.word_index
total_words=len(word_index)+1
#corpus=tokenizer.text_to_sequences(corpus)

#print(total_words)
input_sequences=[]
for line in corpus:
    token_list=tokenizer.texts_to_sequences([line])[0]#will give sequence of one line
    for i in range(1,len(token_list)):#for each word in line
        n_gram_sequences=token_list[:i+1] #for i=1 frst and second word will be ppended
        input_sequences.append(n_gram_sequences)
#print(input_sequences)
        
import numpy as np
max_len_sequences=max(len(x) for x in input_sequences)
#padded=pad_sequences(input_sequences,maxlen=max_len_sequences,padding="pre")
#print(padded)
input_sequences=np.array(pad_sequences(input_sequences,maxlen=max_len_sequences,padding="pre"))
     
xs,labels= input_sequences[:,:-1],input_sequences[:,-1] 
#print(labels)
ys=tf.keras.utils.to_categorical(labels,num_classes=total_words)
#print(ys[2])

#print(total_words)
embed_dim=32
'''model=tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words,embed_dim,input_length=10) ,
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(4)),
    tf.keras.layers.Dense(total_words,activation="softmax")
])
'''

model=Sequential()
model.add(Embedding(total_words,100,input_length=max_len_sequences-1))#we do -1 bcoz we chop of last word from each sentence above
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words,activation="softmax"))
adam=Adam(lr=0.01)#learning rate=0.01
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
#model.summary()
history=model.fit(xs,ys,epochs=10,verbose=1)

test_text="And therefore never"
num_words=10

for _ in range(num_words):
    token_list=tokenizer.texts_to_sequences([test_text])[0]
    token_list=pad_sequences([token_list],maxlen=max_len_sequences-1,padding="pre")
    predicted=model.predict_classes(token_list,verbose=0)
    output_word=""
    for word,index in word_index.items():
        if index==predicted:
            output_word=word
            break
    test_text+= " " +output_word
    
print(test_text)



