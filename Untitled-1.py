# %%
import numpy as np 
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense

# %%
## LOAd the dataset

max_features= 100000
(X_train,y_train),(X_test,y_test)=imdb.load_data(num_words=max_features)

# %%
print(f'{X_train.shape},{y_train.shape}')

# %%
sample_rev=X_train[0]
sample_label=y_train[0]

# print(f"{sample_rev},{sample_label}")

# %%
# X_train[0]


# %%
word_ind=imdb.get_word_index()
index = word_ind.get("")
print(index)

# %%
# word_ind

reverse_word_ind={value: key for key, value in word_ind.items()}
reverse_word_ind

# %%
# reverse_word_ind

# %%
decoded_review=' '.join([reverse_word_ind.get(i-3,'?') for i in sample_rev])

# %%
decoded_review

# %%
max_len=500
X_train=sequence.pad_sequences(X_train,maxlen=max_len)
X_test=sequence.pad_sequences(X_test,maxlen=max_len)
# X_train[0]

# %%
model=Sequential()
model.add(Embedding(max_features,128,input_length=max_len))
model.add(SimpleRNN(128,activation='tanh'))
model.add(Dense(1,activation="sigmoid"))
model.build(input_shape=(None, 500))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# %%
model.summary()

# %%
from tensorflow.keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)

# %%
history=model.fit(X_train,y_train,epochs=10,batch_size=32,validation_split=0.2,callbacks=[early_stopping])


# %%
gpus = tf.config.list_physical_devices('GPU')

# Print the number of GPUs available
print("Num GPUs Available: ", len(gpus))

# %%
model.save('simple_rnn_imdb.h5')

# %%
import matplotlib.pyplot as plt


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%


# %%



