import os
import json
import math
import matplotlib.pyplot as plt
from data_processor import DataProcessor
from model import Model

def plot_results(prediction, actual, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(actual, label='Actual')
	# Pad the list of predictions to shift it to its correct start
    for i, data in enumerate(prediction):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


if __name__ == '__main__':
    configuration = json.load(open('config.json', 'r'))
    if not os.path.exists(configuration['model']['save_dir']): os.makedirs(configuration['model']['save_dir'])

    data = DataProcessor(
        os.path.join('data', configuration['data']['filename']),
        configuration['data']['train_test_split'],
        configuration['data']['columns']
    )

    model = Model()
    model.build_model(configuration)
    x, y = data.get_train_data(
        seq_len = configuration['data']['sequence_length'],
        normalize = configuration['data']['normalize']
    )

    # out-of memory generative training
    steps_per_epoch = math.ceil((data.len_train - configuration['data']['sequence_length']) / configuration['training']['batch_size'])
    model.train_generator(
        data_gen = data.generate_train_batch(
            seq_len = configuration['data']['sequence_length'],
            batch_size = configuration['training']['batch_size'],
            normalize = configuration['data']['normalize']
        ),
        epochs = configuration['training']['epochs'],
        batch_size = configuration['training']['batch_size'],
        steps_per_epoch = steps_per_epoch,
        save_dir = configuration['model']['save_dir']
    )

    x_test, y_test = data.get_test_data(
        seq_len = configuration['data']['sequence_length'],
        normalize = configuration['data']['normalize']
    )

    predictions = model.predict_sequences_multiple(x_test, configuration['data']['sequence_length'], configuration['data']['sequence_length'])

    plot_results(predictions, y_test, configuration['data']['sequence_length'])
