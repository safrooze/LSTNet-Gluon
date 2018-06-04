from __future__ import print_function

import argparse

import mxnet as mx
from mxnet import nd, gluon, autograd

from dataset import TimeSeriesData
from model import LSTNet


def train(file_path, out_path):
    ts_data = TimeSeriesData(file_path, window=24*7, horizon=24)

    ctx = mx.gpu(0)

    net = LSTNet(
        num_series=ts_data.num_series,
        conv_hid=100,
        gru_hid=100,
        skip_gru_hid=5,
        skip=24,
        ar_window=24)
    l1 = gluon.loss.L1Loss()

    net.initialize(init=mx.init.Xavier(factor_type="in", magnitude=2.34), ctx=ctx)

    trainer = gluon.Trainer(net.collect_params(),
                            optimizer='adam',
                            optimizer_params={'learning_rate': 0.001, 'clip_gradient': 10.})

    batch_size = 128
    train_data_loader = gluon.data.DataLoader(
        ts_data.train, batch_size=batch_size, shuffle=True, num_workers=16, last_batch='discard')

    scale = nd.array(ts_data.scale, ctx=ctx)
    epochs = 100
    loss = None
    print("Training Start")
    for e in range(epochs):
        epoch_loss = mx.nd.zeros((1,), ctx=ctx)
        num_iter = 0
        for data, label in train_data_loader:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            if loss is not None:
                loss.wait_to_read()
            with autograd.record():
                y_hat = net(data)
                loss = l1(y_hat * scale, label * scale)
            loss.backward()
            trainer.step(batch_size)
            epoch_loss = epoch_loss + loss.mean()
            num_iter += 1
        print("Epoch {:3d}: loss {:.4}".format(e, epoch_loss.asscalar() / num_iter))

    net.save_params(out_path)
    print("Training End")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTNet Time series forecasting')
    parser.add_argument('--data', type=str, required=True,
                        help='path of the data file')
    parser.add_argument('--out', type=str, required=True,
                        help='path of the trained network output')
    args = parser.parse_args()

    exit(train(args.data, args.out))
