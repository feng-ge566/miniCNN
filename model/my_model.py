import logging

from .my_CNN_cifar10 import vgg16 as my_CNN_cifar10


def create_model(args,select_dict=None):
    logger = logging.getLogger()

    model = None


    if args.arch == 'my_CNN_cifar10':
        model = my_CNN_cifar10(pretrained=args.pre_trained)
    else:
        print("no arch name %s \n", args.arch)

    if model is None:
        logger.error('Model architecture `%s` for `%s` dataset is not supported' % (args.arch, args.dataloader.dataset))
        exit(-1)

    msg = 'Created `%s` model for `imagenet` dataset' % (args.arch)
    msg += '\n          Use pre-trained model = %s' % args.pre_trained
    logger.info(msg)

    return model
