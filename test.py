import filemanager
import keras
import tensorflow as tf


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
model = keras.models.load_model('D:\\leguan_data\\leguan_models\\leguan_model_1.2.hdf5')

f1, f2 = filemanager.open_files(3)
gen = filemanager.batch_generator(128, f1, f2)


x, y_true = next(gen)



x = tf.reshape(x, (1, -1, 12, 8, 8))

y_predict = model.predict(tuple(x))

print(y_true)

print(y_predict)


