from pandas import read_csv
import tensorflow as tf


def load_translate_from_csv(filepath, sep=";", val_rate=0.1):
    df = read_csv(filepath, sep=sep)
    df_list = list(df.itertuples(index=False, name=None))
    dataset_size = len(df_list)
    val_size = int(dataset_size * val_rate)
    train_size = dataset_size - val_size
    dataset = tf.data.Dataset.from_tensor_slices(df_list)
    dataset = dataset.map(lambda x: tf.unstack(x))

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    return {"train": train_dataset, "validation": val_dataset}


def print_translation(sentence, tokens, ground_truth):
  print(f'{"Input:":15s}: {sentence}')
  print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
  print(f'{"Ground truth":15s}: {ground_truth}')
