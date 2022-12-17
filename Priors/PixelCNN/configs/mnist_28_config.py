config = {}
config["batch_size"] = 128          # input batch size for training (default: 64)
config["epochs"] = 100             # number of epochs to train (default: 10)
config["no_cuda"] = False         # disables CUDA training
config["seed"] = 1265
config["log_interval"] = 1     # how many batches to wait before logging training status
config["learning_rate"] = 1e-3
config["momentum"] = 0.1
config["gamma"] = 0.99
config["weight_decay"] = 0.0

config["num_hiddens"] = 64
config["num_channels"] = 1
config["data_set"] = "MNIST"
config["image_size"] = 28
config["representation_dim"] = 28
config["num_categories"] = 256
