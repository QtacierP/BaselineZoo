from torch.optim import Adam, SGD, Adadelta, Adagrad

optimizer_list = {
    'adam': Adam, 
    'sgd': SGD,
    'adadelta': Adadelta,
    'adagrad': Adagrad}