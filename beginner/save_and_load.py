import os

import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


# model = create_model()

checkpoint_path = './training_1/cp.ckpt'
# checkpoint_dir = os.path.dirname(checkpoint_path)
#
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_best_only=True,
#                                                  verbose=1)
#
# model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels),
#           callbacks=[cp_callback])

# model = create_model()
#
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print("Accuracy of untrained model: {:5.2f}%".format(100*acc))
#
# model.load_weights(checkpoint_path)
#
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print("Accuracy of trained model: {:5.2f}%".format(100*acc))
#

# checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
#
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path,
#     verbose=1,
#     save_best_only=True,
#     period=5
# )
#
# model = create_model()
#
# model.save(checkpoint_path.format(epoch=0))
#
# model.fit(train_images,
#           train_labels,
#           epochs=50,
#           callbacks=[cp_callback],
#           validation_data=(test_images, test_labels),
#           verbose=0)

# model = create_model()
# model.fit(train_images, train_labels, epochs=5)
#
# model.save('saved_model/my_model')

# new_model = tf.keras.models.load_model('./saved_model/my_model')
#
# loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
#
# print('복원된 모델의 정확도: {:5.2f}%'.format(100*acc))
#
# print(new_model.predict(test_images).shape)