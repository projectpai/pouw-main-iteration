from _jsonnet import evaluate_file
import yaml
import random

from pai.pouw.mining.blkmaker.blkmaker import sha256_hexdigest
from pai.pouw.dkg.joint_feldman import message

version = 0.1
best_local_payment_amount = 0.3
ml_dataset_format = ['MNIST', 'CSV']

validation_method = ["Holdout", "k-fold", "Leave-one-out"]
validation_holdout_pct = 0.1


if __name__ == "__main__":
    # seed the random number generator
    seed = sha256_hexdigest(message.encode())
    random.seed(seed)

    ext_vars = {'ml_dataset_format': random.choice(ml_dataset_format),
                'validation_method': random.choice(validation_method)}

    test_set_size = random.uniform(0.2, 0.4)

    ext_codes = {'version': str(version),
                 'best_local_payment_amount': str(best_local_payment_amount),
                 'validation_holdout_pct': str(validation_holdout_pct),
                 'epochs': str(random.randint(10, 50)),
                 'tau': str(random.randint(1, 20)),
                 'learning_rate': str(random.uniform(0.1, 0.9)),
                 'momentum': str(random.uniform(0.0, 0.8)),
                 'batch_size': str(random.choice([16, 64, 128, 256])),
                 'xavier_magnitude': str(random.uniform(2.0, 3.0)),
                 'test_set_size': str(test_set_size),
                 'training_set_size': str(1-test_set_size)}

    raw_str = evaluate_file("./template.jsonnet", fp=2,
                            ext_vars=ext_vars, ext_codes=ext_codes)
    yml = yaml.load(raw_str, yaml.UnsafeLoader)
    print(yml)