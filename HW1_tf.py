import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

import datetime

lossgraph = False
hist = keras.callbacks.History()

mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

x_v = train_images[-10000:]
y_v = train_labels[-10000:]

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

hist = model.fit(train_images, train_labels, batch_size=16*16, epochs=5, validation_data=(x_v,y_v), validation_batch_size=16)

if lossgraph:
    fig, loss_ax = plt.subplots()

    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'b', label='validation loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')

    
    acc_ax.set_ylabel('accuracy')
    acc_ax.plot(hist.history['val_accuracy'], 'r', label ='Validation Accuracy')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    plt.show()


predictions = model.predict(test_images)

y_pred = model.predict_classes(test_images)

confusions = tf.math.confusion_matrix(labels=test_labels, predictions=y_pred)

print(type(confusions))


anw = tf.math.argmax(predictions, axis=1)

prob_max = np.array(np.zeros((10,10000)),dtype='f8')
prob_max = np.sort(predictions,axis=1)
prob_top = (prob_max[:,9])

a_0 = []
a_1 = []
a_2 = []
a_3 = []
a_4 = []
a_5 = []
a_6 = [] 
a_7 = []
a_8 = []
a_9 = []

for i in range(len(anw)):
    if anw[i] == 0:
        a_0.append(i)
    elif anw[i] == 1:
        a_1.append(i)
    elif anw[i] == 2:
        a_2.append(i)
    elif anw[i] == 3:
        a_3.append(i)
    elif anw[i] == 4:
        a_4.append(i)
    elif anw[i] == 5:
        a_5.append(i)
    elif anw[i] == 6:
        a_6.append(i)
    elif anw[i] == 7:
        a_7.append(i)
    elif anw[i] == 8:
        a_8.append(i)
    elif anw[i] == 9:
        a_9.append(i)

    if i % 1000 == 0:
        print(str(i) + str(" / 10000 : doing now..."))



# pa_0 = []
pa_0 = []
pa_0_idx = []
print("making 0's probability array",end='')
for i in range(len(a_0)):
    pa_0.append(prob_top[a_0[i]])
    pa_0_idx.append(a_0[i])
print(np.shape(pa_0))

pa_1 = []
pa_1_idx = []
print("making 1's probability array",end='')
for i in range(len(a_1)):
    pa_1.append(prob_top[a_1[i]])
    pa_1_idx.append(a_1[i])
print(np.shape(pa_1))

pa_2 = []
pa_2_idx = []
print("making 2's probability array",end='')
for i in range(len(a_2)):
    pa_2.append(prob_top[a_2[i]])
    pa_2_idx.append(a_2[i])
print(np.shape(pa_2))

pa_3 = []
pa_3_idx = []
print("making 3's probability array",end='')
for i in range(len(a_3)):
    pa_3.append(prob_top[a_3[i]])
    pa_3_idx.append(a_3[i])
print(np.shape(pa_3))

pa_4 = []
pa_4_idx = []
print("making 4's probability array",end='')
for i in range(len(a_4)):
    pa_4.append(prob_top[a_4[i]])
    pa_4_idx.append(a_4[i])
print(np.shape(pa_4))

pa_5 = []
pa_5_idx = []
print("making 5's probability array",end='')
for i in range(len(a_5)):
    pa_5.append(prob_top[a_5[i]])
    pa_5_idx.append(a_5[i])
print(np.shape(pa_5))

pa_6 = [] 
pa_6_idx = []
print("making 6's probability array",end='')
for i in range(len(a_6)):
    pa_6.append(prob_top[a_6[i]])
    pa_6_idx.append(a_6[i])
print(np.shape(pa_6))

pa_7 = []
pa_7_idx = []
print("making 7's probability array",end='')
for i in range(len(a_7)):
    pa_7.append(prob_top[a_7[i]])
    pa_7_idx.append(a_7[i])
print(np.shape(pa_7))

pa_8 = []
pa_8_idx = []
print("making 8's probability array",end='')
for i in range(len(a_8)):
    pa_8.append(prob_top[a_8[i]])
    pa_8_idx.append(a_8[i])
print(np.shape(pa_8))

pa_9 = []
pa_9_idx = []
print("making 9's probability array",end='')
for i in range(len(a_9)):
    pa_9.append(prob_top[a_9[i]])
    pa_9_idx.append(a_9[i])
print(np.shape(pa_9))

top_3_0 = np.argsort(pa_0)[-3:]
top_3_1 = np.argsort(pa_1)[-3:]
top_3_2 = np.argsort(pa_2)[-3:]
top_3_3 = np.argsort(pa_3)[-3:]
top_3_4 = np.argsort(pa_4)[-3:]
top_3_5 = np.argsort(pa_5)[-3:]
top_3_6 = np.argsort(pa_6)[-3:]
top_3_7 = np.argsort(pa_7)[-3:]
top_3_8 = np.argsort(pa_8)[-3:]
top_3_9 = np.argsort(pa_9)[-3:]

top_0_ids = [ pa_0_idx[top_3_0[0]], pa_0_idx[top_3_0[1]], pa_0_idx[top_3_0[2]]]
top_1_ids = [pa_1_idx[top_3_0[0]],pa_1_idx[top_3_1[1]],pa_1_idx[top_3_1[2]]]
top_2_ids = [pa_2_idx[top_3_0[0]],pa_2_idx[top_3_2[1]],pa_2_idx[top_3_2[2]]]
top_3_ids = [pa_3_idx[top_3_0[0]],pa_3_idx[top_3_3[1]],pa_3_idx[top_3_3[2]]]
top_4_ids = [pa_4_idx[top_3_0[0]],pa_4_idx[top_3_4[1]],pa_4_idx[top_3_4[2]]]
top_5_ids = [pa_5_idx[top_3_0[0]],pa_5_idx[top_3_5[1]],pa_5_idx[top_3_5[2]]]
top_6_ids = [pa_6_idx[top_3_0[0]],pa_6_idx[top_3_6[1]],pa_6_idx[top_3_6[2]]]
top_7_ids = [pa_7_idx[top_3_0[0]],pa_7_idx[top_3_7[1]],pa_7_idx[top_3_7[2]]]
top_8_ids = [pa_8_idx[top_3_0[0]],pa_8_idx[top_3_8[1]],pa_8_idx[top_3_8[2]]]
top_9_ids = [pa_9_idx[top_3_0[0]],pa_9_idx[top_3_9[1]],pa_9_idx[top_3_9[2]]]

top_idx = []

top_idx.append(top_0_ids)
top_idx.append(top_1_ids)
top_idx.append(top_2_ids)
top_idx.append(top_3_ids)
top_idx.append(top_4_ids)
top_idx.append(top_5_ids)
top_idx.append(top_6_ids)
top_idx.append(top_7_ids)
top_idx.append(top_8_ids)
top_idx.append(top_9_ids)

top_idx = np.squeeze(np.reshape(top_idx,(30,1)))

print("0's top 3 index in test set is " + str(pa_0_idx[top_3_0[0]]) + ', ' + str(pa_0_idx[top_3_0[1]]) + ', ' + str(pa_0_idx[top_3_0[2]]))
print("1's top 3 index in test set is " + str(pa_1_idx[top_3_1[0]]) + ', ' + str(pa_1_idx[top_3_1[1]]) + ', ' + str(pa_1_idx[top_3_1[2]]))
print("2's top 3 index in test set is " + str(pa_2_idx[top_3_2[0]]) + ', ' + str(pa_2_idx[top_3_2[1]]) + ', ' + str(pa_2_idx[top_3_2[2]]))
print("3's top 3 index in test set is " + str(pa_3_idx[top_3_3[0]]) + ', ' + str(pa_3_idx[top_3_3[1]]) + ', ' + str(pa_3_idx[top_3_3[2]]))
print("4's top 3 index in test set is " + str(pa_4_idx[top_3_4[0]]) + ', ' + str(pa_4_idx[top_3_4[1]]) + ', ' + str(pa_4_idx[top_3_4[2]]))
print("5's top 3 index in test set is " + str(pa_5_idx[top_3_5[0]]) + ', ' + str(pa_5_idx[top_3_5[1]]) + ', ' + str(pa_5_idx[top_3_5[2]]))
print("6's top 3 index in test set is " + str(pa_6_idx[top_3_6[0]]) + ', ' + str(pa_6_idx[top_3_6[1]]) + ', ' + str(pa_6_idx[top_3_6[2]]))
print("7's top 3 index in test set is " + str(pa_7_idx[top_3_7[0]]) + ', ' + str(pa_7_idx[top_3_7[1]]) + ', ' + str(pa_7_idx[top_3_7[2]]))
print("8's top 3 index in test set is " + str(pa_8_idx[top_3_8[0]]) + ', ' + str(pa_8_idx[top_3_8[1]]) + ', ' + str(pa_8_idx[top_3_8[2]]))
print("9's top 3 index in test set is " + str(pa_9_idx[top_3_9[0]]) + ', ' + str(pa_9_idx[top_3_9[1]]) + ', ' + str(pa_9_idx[top_3_9[2]]))

# print(top_3_0)
# print(top_3_1)
# print(top_3_2)
# print(top_3_3)
# print(top_3_4)
# print(top_3_5)
# print(top_3_6)
# print(top_3_7)
# print(top_3_8)
# print(top_3_9)

print(top_idx)

plt.figure(figsize=(10,10))
for i in range(30):
    plt.subplot(10,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[top_idx[i]], cmap=plt.cm.binary)
    plt.xlabel(test_labels[top_idx[i]])
plt.show()
