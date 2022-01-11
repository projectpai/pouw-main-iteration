import pandas as pd
import time
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, RobustScaler
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


def load_csv(file_name, data_exceptions, labels):
    x_all = pd.read_csv(file_name)

    y_all = x_all[labels]
    data_exceptions.extend(labels)
    for data_ex in data_exceptions:
        del x_all[data_ex]

    np_features = x_all.to_numpy()
    features_scaler = RobustScaler()
    features_scaler.fit(np_features)
    np_features = features_scaler.transform(np_features)

    np_labels = y_all.to_numpy()
    labels_scaler = RobustScaler()
    labels_scaler.fit(np_labels)
    np_labels = labels_scaler.transform(np_labels)
    x_train, x_test, y_train, y_test = train_test_split(np_features, np_labels, test_size=0.3, random_state=42)
    return (x_train, y_train), (x_test, y_test)


def train():
    # Get model
    inputs = keras.Input(shape=(94,), name="inputs")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(4, name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Instantiate an optimizer to train the model.
    optimizer = keras.optimizers.SGD(learning_rate=1e-3)
    # Instantiate a loss function.
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Prepare the metrics.
    train_mse_metric = tf.keras.metrics.RootMeanSquaredError()
    val_acc_metric = tf.keras.metrics.RootMeanSquaredError()

    # Prepare the training dataset.
    batch_size = 32
    (x_train, y_train), (x_test, y_test) = load_csv('data_clean.csv', ["Filename", "docket_number", "remanded", "result text"], ['result'])

    val_size = 100
    # Reserve 10,000 samples for validation.
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=128).batch(batch_size)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)

    epochs = 30
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            train_mse_metric.update_state(y_batch_train, logits)

            # Log every 10 batches.
            if step % 100 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %d samples" % ((step + 1) * batch_size))

        # Display metrics at the end of each epoch.
        train_metric = train_mse_metric.result()
        print("Training MSE over epoch: %.4f" % (float(train_metric),))

        # Reset training metrics at the end of each epoch
        train_mse_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation MSE: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))


if __name__ == '__main__':
    # load_csv('data_clean.csv')
    train()
