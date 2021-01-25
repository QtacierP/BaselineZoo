
def process_args(args):
    args.gpu = [int(gpu) for gpu in args.gpu.split(',')]
    if len(args.gpu) == 1:
        args.accelerator = None
    return args