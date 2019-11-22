std.manifestYamlDoc({
    "ml": {
    "evaluation-metrics": [
      "Accuracy",
      "CrossEntropy"
    ],
    "model": {
      "loss": "SoftmaxCrossEntropyLoss",
      "type": "FC-DNN",
      "hidden-units": [
        {
          "nodes": 128,
          "type": "Dense",
          "activation": "relu",
          "id": "layer1"
        },
        {
          "nodes": 64,
          "type": "Dense",
          "activation": "relu",
          "id": "layer2"
        },
        {
          "nodes": 10,
          "type": "Dense",
          "id": "output"
        }
      ]
    },
    "validation": {
      "strategy": {
        "method": std.extVar('validation_method'),
        "size": std.extVar('validation_holdout_pct')
      }
    },
    "optimizer": {
      "epochs": std.extVar('epochs'),
      "tau": std.extVar('tau'),
      "optimizer_initialization_parameters": {
        "learning_rate": std.extVar('learning_rate'),
        "momentum": std.extVar('momentum')
      },
      "batch-size": std.extVar('batch_size'),
      "type": "SGD",
      "initializer": {
        "name": "Xavier",
        "parameters": {
          "magnitude": std.extVar('xavier_magnitude')
        }
      }
    },
    "dataset": {
      "source": {
        "labels": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "features": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
      },
      "test-set-size": std.extVar('test_set_size'),
      "sha256": "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
      "training-set-size": std.extVar('training_set_size'),
      "format": "MNIST"
    }
  },
  "version": std.extVar('version'),
  "payment": {
    "best-model-payment-amount": std.extVar('best_local_payment_amount')
  }
})
