# code adaptation from https://keras.io/guides/writing_a_training_loop_from_scratch/

import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from pai.pouw.nodes.decentralized.message_map import rebuild_delta_local


def is_upper_threshold(var):
    return var & (1 << 31) > 0


class Trainer:
    def __init__(self):
        self._residualGradients = None
        self._tau = 10.00
        # add the default MNIST model

        inputs = keras.Input(shape=(784,), name="digits")
        x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
        x = layers.Dense(64, activation="relu", name="dense_2")(x)
        outputs = layers.Dense(10, name="predictions")(x)
        self._model = keras.Model(inputs=inputs, outputs=outputs)

        # set the default optimizer
        self._optimizer = keras.optimizers.SGD(learning_rate=1e-3)

        # store structure
        self._structure = [[w.shape.dims[0].value] if w.shape.ndims == 1 else [w.shape.dims[0].value, w.shape.dims[1].value] for w in self.model.trainable_weights]

        # store ranges for local map
        self._ranges = []
        cumulated = 0
        for itm in self.structure:
            prev = cumulated
            cumulated += (itm[0] if len(itm) == 1 else itm[0] * itm[1])
            self._ranges.append((prev, cumulated - 1))

        # set the loss function
        self._loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # prepare the metrics
        self._train_metric = keras.metrics.SparseCategoricalAccuracy()
        self._val_metric = keras.metrics.SparseCategoricalAccuracy()

        self._model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=[self.train_metric])

        # set the batch size
        self._batch_size = 64

        # datasets
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self.prepare_datasets()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, value):
        self._structure = value

    @property
    def ranges(self):
        return self._ranges

    @ranges.setter
    def ranges(self, value):
        self._ranges = value

    @property
    def train_dataset(self):
        return self._train_dataset

    @train_dataset.setter
    def train_dataset(self, value):
        self._train_dataset = value

    @property
    def val_dataset(self):
        return self._val_dataset

    @val_dataset.setter
    def val_dataset(self, value):
        self._val_dataset = value

    @property
    def test_dataset(self):
        return self._test_dataset

    @test_dataset.setter
    def test_dataset(self, value):
        self._test_dataset = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def loss_fn(self):
        return self._loss_fn

    @loss_fn.setter
    def loss_fn(self, value):
        self.loss_fn = value

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def train_metric(self):
        return self._train_metric

    @train_metric.setter
    def train_metric(self, value):
        self._train_metric = value

    @property
    def val_metric(self):
        return self._val_metric

    @val_metric.setter
    def val_metric(self, value):
        self._val_metric = value

    @property
    def residual_gradients(self):
        return self._residualGradients

    @residual_gradients.setter
    def residual_gradients(self, value):
        self._residualGradients = value

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, value):
        self._tau = value

    def prepare_datasets(self):
        # Prepare the training dataset.
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.reshape(x_train, (-1, 784))
        x_test = np.reshape(x_test, (-1, 784))

        self.test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        # prepare the validation dataset
        x_val = x_train[-10000:]
        y_val = y_train[-10000:]
        x_train = x_train[:-10000]

        y_train = y_train[:-10000]
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        self.val_dataset = val_dataset.batch(self.batch_size)

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self.train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)

    def loop(self):
        epochs = 5
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
                with open('/test/x_batch_train.npy', 'wb') as x_val_f,\
                        open('/test/y_batch_train.npy', 'wb') as y_val_f:
                    np.save(x_val_f, x_batch_train.numpy())
                    np.save(y_val_f, y_batch_train.numpy())

                x_val = np.load('/test/x_batch_train.npy')
                y_val = np.load('/test/y_batch_train.npy')

                for x_train_2, y_train_2 in tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(64):
                    test1 = tf.reduce_all(tf.equal(x_batch_train, x_train_2))
                    test2 = tf.reduce_all(tf.equal(y_batch_train, y_train_2))
                    print(test1)
                #tf.data.experimental.save(self.train_dataset, '/Volumes/blackbox/Archive/uar/pouw-main-iteration/test/saved_batch_x')
                # tf.io.write_file('/Volumes/blackbox/Archive/uar/pouw-main-iteration/test/saved_batch_y', x_batch_train)

                with tf.GradientTape() as tape:
                    logits = self.model(x_batch_train, training=True)
                    loss_value = self.loss_fn(y_batch_train, logits)
                grads = tape.gradient(loss_value, self.model.trainable_weights)

                self.add_new_gradients_to_residuals(grads)
                delta_local, map_local = self.compute_delta_local_and_update_residuals()
                delta_local_reconstructed = rebuild_delta_local(map_local, self.model.trainable_weights,
                                                                self.tau, self.structure, self.ranges)
                for idx, el in enumerate(delta_local):
                    t1 = el.numpy()
                    t2 = delta_local_reconstructed[idx].numpy()
                    res = np.array_equal(t1, t2)
                    print(res)
                self.optimizer.apply_gradients(zip(delta_local, self.model.trainable_weights))

                # Update training metric.
                self.train_metric.update_state(y_batch_train, logits)

                # Log every 200 batches.
                if step % 200 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %d samples" % ((step + 1) * 64))

            # Display metrics at the end of each epoch.
            train_acc = self.train_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))

            # Reset training metrics at the end of each epoch
            self.train_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in self.val_dataset:
                val_logits = self.model(x_batch_val, training=False)
                # Update val metrics
                self.val_metric.update_state(y_batch_val, val_logits)
            val_acc = self.val_metric.result()
            self.val_metric.reset_states()
            print("Validation acc: %.4f" % (float(val_acc),))
            print("Time taken: %.2fs" % (time.time() - start_time))

    def compute_delta_local_and_update_residuals(self):
        delta_local = []
        map_local = []
        address = 0
        for idx, res_grad in enumerate(self.residual_gradients):
            upper_grads = tf.math.greater_equal(self.residual_gradients[idx], self.tau)
            lower_grads = tf.math.less_equal(self.residual_gradients[idx], -self.tau)
            map_local.extend(address + (int(up_idx[0] if res_grad.shape.ndims == 1 else up_idx[0] * res_grad.shape[1] + up_idx[1]) | (1 << 31)) for up_idx in tf.where(tf.equal(upper_grads, True)))
            map_local.extend(address + (int(lw_idx[0] if res_grad.shape.ndims == 1 else lw_idx[0] * res_grad.shape[1] + lw_idx[1]) & (~(1 << 31))) for lw_idx in tf.where(tf.equal(lower_grads, True)))
            address += res_grad.shape[0] if res_grad.shape.ndims == 1 else res_grad.shape[0] * res_grad.shape[1]
            delta_local_row = tf.cast(upper_grads, tf.float32) * self.tau + tf.cast(lower_grads, tf.float32) * -self.tau
            self.residual_gradients[idx] = tf.math.subtract(self.residual_gradients[idx], delta_local_row)
            delta_local.append(delta_local_row)

        return delta_local, map_local

    def add_new_gradients_to_residuals(self, grads):
        if not self.residual_gradients:
            self.residual_gradients = [tf.identity(grad) for grad in grads]
        else:
            for idx, grad in enumerate(grads):
                self.residual_gradients[idx] = tf.math.add(self.residual_gradients[idx], grad)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.loop()
