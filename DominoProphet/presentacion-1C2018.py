import glob
from PIL import Image
import numpy as np
import json
import tensorflow as tf
import sys
import pandas as pd

#CANTIDAD DE PIEZAS (CATEGORIAS) DEL DATASET
PIECES = 28

#TAMANO DE LAS IMAGENES EN PX
IMAGE_HEIGHT = 45
IMAGE_WIDTH = 45

TOTAL_PIXELS = IMAGE_WIDTH*IMAGE_HEIGHT

# Nombre del modelo pre entrenado:
model_name = "model_0.656746"

# True para usar el modelo pre entrenado, False para entrenar
USING_LOADED_MODEL = False

# True para usar el json que represenra las imagenes ya cread, False para volver a generar el json de train y test.
USING_EXISTING_DATASET_JSONS = False

# True para invertir los colores, lo blanco a negro, lo negro a blanco
INVERT_COLORS = False

# Para pruebas deterministicas descomentar
# np.random.seed(0)

def load_image(addr):
    img = Image.open(addr)
    img = img.convert('L')
    img = np.array(img).flatten().astype(np.int)
    if INVERT_COLORS:
        inverse = lambda t: 255-t
        invert = np.vectorize(inverse)
        img = invert(img)
    img = img.tolist()
    return img

def load_label(addr):
    label = np.array(int(addr.split("-")[0].split("/")[2].split("_")[0]))
    ha_label = np.zeros((label.size, PIECES))
    ha_label[np.arange(label.size), label] = 1
    ha_label = ha_label[0].astype(np.int).tolist()
    return ha_label

def load_images_dataset(addrs):
    images = {}
    for i in range(len(addrs)):
        addr = addrs[i]
        image = {'image': load_image(addr), 'label': load_label(addr)}
        images[i] = image
    return images

def write_images_dataset(dataset, file):
    try:
        with open(file,'w') as outfile:
            json.dump(dataset, outfile, indent=4)
    except:
        print("Error", sys.exc_info()[0])
        quit()
    return

def read_images_dataset(file):
    try:
        with open(file) as in_file:    
            data = json.load(in_file)
        labels = np.array([data[str(i)]['label'] for i in range(len(data))])
        images = np.array([data[str(i)]['image'] for i in range(len(data))])
        dataset = {'images': images, 'labels': labels}
    except:
        print("Error:", sys.exc_info()[0])
        quit()
    return dataset

def batch_data(source, target, batch_size):
    shuffle_indices = np.random.permutation(np.arange(len(target)))
    source = source[shuffle_indices]
    target = target[shuffle_indices]

    for batch_i in range(0, len(source)//batch_size):
        start_i = batch_i * batch_size
        source_batch = source[start_i:start_i + batch_size]
        target_batch = target[start_i:start_i + batch_size]
        yield np.array(source_batch), np.array(target_batch)

# Cargamos el training set
dominos_train_path = 'images/trainingImages/*.jpg'
json_file = 'trainDataSet.json'
if not USING_EXISTING_DATASET_JSONS:
    # Levantamos las imagenes y generamos un json con su representacion
    print("Generando json con train data...")
    img_addrs = sorted(glob.glob(dominos_train_path))
    img_dataset = load_images_dataset(img_addrs)
    write_images_dataset(img_dataset, json_file)
# Levantamos el json con la representacion de las imagenes
print("Cargando json con train data...")
dict_dataset_train = read_images_dataset(json_file)

# Cargamos el validation set
dominos_test_path = 'images/testImages/*.jpg'
json_file = 'testDataSet.json'
if not USING_EXISTING_DATASET_JSONS:
    # Levantamos las imagenes y generamos un json con su representacion
    print("Generando json con test data...")
    img_addrs = sorted(glob.glob(dominos_test_path))
    img_dataset = load_images_dataset(img_addrs)
    write_images_dataset(img_dataset, json_file)
# Levantamos el json con la representacion de las imagenes
print("Cargando json con test data...")
dict_dataset_test = read_images_dataset(json_file)

x = tf.placeholder(tf.float32, [None, TOTAL_PIXELS])

# Modelo a entrenar
W = tf.Variable(tf.zeros([TOTAL_PIXELS, PIECES]))
b = tf.Variable(tf.zeros([PIECES]))

y_ = tf.placeholder(tf.float32, [None, PIECES])

y = tf.matmul(x, W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver({"W": W, "b": b})

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

if (USING_LOADED_MODEL == False):
    BATCH_SIZE = 150
    batch_train = batch_data(dict_dataset_train['images'],dict_dataset_train['labels'],BATCH_SIZE)

if (USING_LOADED_MODEL == False):
    z = tf.matmul(x, W) + b
    size = len(dict_dataset_train['images'])
    for i in range((size//BATCH_SIZE)):
        batch = next(batch_train)
        batch_xs = batch[0]
        batch_ys = batch[1]
        _, loss_val, W_val, b_val, z_val = sess.run([train_step, cross_entropy, W, b, z],
                                      feed_dict={x: batch_xs, y_: batch_ys})
# Valores durante el entrenamiento:
#         print(z_val)
#         print('loss =', loss_val)
#         print('W =', W_val)
#         print('b =', b_val)


if USING_LOADED_MODEL:
    saver.restore(sess, model_name+".ckpt")
    print("Cargado modelo pre entrenado.")

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

model_accuracy = sess.run(accuracy, feed_dict={x: dict_dataset_test['images'], y_: dict_dataset_test['labels']})
# Imprimimos la presicion del modelo con el set de test:
print("La presicion del modelo es:", model_accuracy*100, "%")


# Generamos un archivo con la prediccion de cada elemento del validation set para su analisis.
argmax = lambda t: t.argmax()
originals = np.array(list(map(argmax, dict_dataset_test['labels'])))
prediction = tf.argmax(y,1)
predictions = prediction.eval(feed_dict={x: dict_dataset_test['images']}, session=sess)
pairs = np.array((originals,predictions, originals==predictions))
pd.DataFrame(pairs).T.to_csv('test-prediction.csv', sep='\t', encoding='utf-8')


if (USING_LOADED_MODEL == False):
    save_path = saver.save(sess, "model_"+str(model_accuracy)+".ckpt")
    print("Modelo guardado: %s" % save_path)

