
def process_args(args):
    args.gpu = [int(gpu) for gpu in args.gpu.split(',')]
    return args