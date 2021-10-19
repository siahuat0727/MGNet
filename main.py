import os
from datetime import datetime
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


import data_modules as dms
from litmodule import LitModule
from utils.callbacks import FlopCount, ValVisualize


def get_callbacks(hparams):
    callbacks = [LearningRateMonitor()]
    if hparams.visualize:
        callbacks.append(ValVisualize(plot_per_val=hparams.img_per_val))
    if hparams.flop_count:
        callbacks.append(FlopCount())
    return callbacks


def get_datamodule(hparams):
    return {
        'mnist_toy': dms.MNIST_toy,
        'mnist': dms.MNIST,
        'imagenet': dms.ImageNet,
    }[hparams.dataset](hparams)


def get_name(hparams):
    return '_'.join(filter(None, [  # Remove empty string by filtering
        f'{datetime.now().strftime("%m%d_%H%M")}',
        f'b{hparams.batch_size}x{hparams.world_size}',
        f'i{hparams.n_iter}s{hparams.scale}',
        f'scale{[hparams.scale_min, hparams.scale_max]}',
        f'{hparams.optimizer}{hparams.learning_rate}',
        f'n{hparams.n_iter_coef}' if hparams.step_by_step is not None else 'sbs',
        f'clip{hparams.clip}' if hparams.clip > 0.0 else '',
        f'gr{hparams.s}' if hparams.s != 1.0 else '',
        f'ge{hparams.ge}' if hparams.ge is not None else '',
        f'{hparams.ge_final}' if hparams.ge_final != hparams.ge else '',
        f'aux{hparams.alpha}' if hparams.aux else '',
        f'ss{hparams.ss_coef}' if hparams.ss else '',
        f'ssl{hparams.ssl_coef}_{hparams.ssl_explore}' if hparams.ssl else '',
        f'div{hparams.div_coef}' if hparams.div else '',
        f'drop{hparams.dropout}' if hparams.dropout != 0.0 else '',
        'noclue' if hparams.no_spatial_clue else '',
        'nodetach' if hparams.no_detach else '',
        f'{hparams.tag}' if hparams.tag is not None else '',
    ])).replace(' ', '')


def main(hparams):
    print(hparams)

    seed_everything(hparams.seed)

    # If only train on 1 GPU.
    if hparams.gpus is not None:
        if hparams.world_size == 1:
            torch.cuda.set_device(int(hparams.gpus.split(',')[0]))

    # Model
    dm = get_datamodule(hparams)
    litmodel = LitModule(hparams, dm)

    name = get_name(hparams)
    logger = TensorBoardLogger(hparams.log_dir, name=name)
    callbacks = get_callbacks(hparams)

    kwargs = {}
    if hparams.world_size > 1:
        kwargs['distributed_backend'] = 'ddp'

    trainer = Trainer(callbacks=callbacks,
                      gpus=hparams.gpus,
                      max_epochs=hparams.epochs,
                      deterministic=True,
                      logger=logger,
                      gradient_clip_val=hparams.clip,
                      check_val_every_n_epoch=hparams.val_per_n,
                      limit_train_batches=hparams.limit_train,
                      limit_val_batches=hparams.limit_val,
                      **kwargs,
                      )

    if hparams.checkpoint is not None:
        load_path = os.path.join(
            hparams.checkpoint, os.listdir(hparams.checkpoint)[0])
        print(f'Loading {load_path} ...')
        litmodel.load_state_dict(torch.load(load_path)['state_dict'])

    if hparams.test:
        trainer.test(litmodel, datamodule=dm)
    else:
        trainer.fit(litmodel, dm)


if __name__ == '__main__':
    parser = ArgumentParser()

    # Dataset
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist_toy', 'mnist', 'imagenet'])
    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--train_dir', type=str, default='data')
    parser.add_argument('--val_dir', type=str, default=None)
    parser.add_argument('--gaussian', type=float, default=0.0,
                        help='Gaussian noise intensity (for mnist)')

    # General
    parser.add_argument('--model', type=str, default='vanilla')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'])
    parser.add_argument('--scheduler', type=str, default='one-cycle',
                        choices=['one-cycle', 'cosine', 'multi-step', 'reduce'])
    parser.add_argument('--schedule', type=int, nargs='+')

    # Environment
    parser.add_argument('--gpus', type=str, default='0,', help='GPUs split with comma')
    parser.add_argument('--workers', type=int, default=4)

    # Training hparams
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Exp. settings
    parser.add_argument('--n_iter', type=int, default=1)
    parser.add_argument('--scale', type=float, default=4)
    parser.add_argument('--scale_min', type=float, default=0.2)
    parser.add_argument('--scale_max', type=float, default=0.5)
    parser.add_argument('--trans_method', type=str, default='tight-crop',
                        choices=['naive', 'tight-crop', 'center-invariant'])
    parser.add_argument('--no_spatial_clue', action='store_true')
    parser.add_argument('--no_detach', action='store_true')

    # Some exploration
    # Auxilary glimpse classifier #
    parser.add_argument('--alpha', type=float, default=0.0)
    # Gradient # Three different methods to modify gradients
    parser.add_argument('--clip', type=float, default=0.0)
    parser.add_argument('--s', type=float, default=1.0,
                        help="Gradient re-scaling factor (localization net)")
    parser.add_argument('--ge', type=float, default=None,
                        help="Dynamic gradient re-scaling factor")
    parser.add_argument('--ge_final', type=float, default=None)
    # Small scale #
    parser.add_argument('--ss_coef', type=float, default=0.0,
            help='A loss to glimpse-region size')
    parser.add_argument('--ss_threshold', type=float, default=0.4)
    # Self-supervised learning (spatial guidance) #
    parser.add_argument('--ssl_coef', type=float, default=0.0,
            help='Self-supervised spatial guidance')
    parser.add_argument('--ssl_explore', type=float, default=0.3)
    # Loss coefficient #
    parser.add_argument('--n_iter_coef', type=int, nargs='+')
    parser.add_argument('--step_by_step', action='store_true')
    # Dropout #
    parser.add_argument('--dropout', type=float, default=0.0)
    # Diverse loss #
    parser.add_argument('--div_coef', type=float, default=0.0)

    # Visualization and logs
    # Visualize #
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--val_per_n', type=int, default=1)
    parser.add_argument('--img_per_val', type=int, default=10)
    # Logs #
    parser.add_argument('--log_dir', type=str, default='logs/')
    parser.add_argument('--tmp', action='store_true')
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    # Debug #
    parser.add_argument('--limit_train', type=float, default=1.0)
    parser.add_argument('--limit_val', type=float, default=1.0)
    # Flop count #
    parser.add_argument('--flop_count', action='store_true',
            help='Count FLOP (use with flag --test)')

    # Adversarial attack
    parser.add_argument('--adv', action='store_true')
    parser.add_argument('--step_k', type=int, default=0)
    parser.add_argument('--step_size', type=float, default=1/255)
    parser.add_argument('--eps', type=float, default=4/255)

    # Reproduce
    parser.add_argument('--seed', type=int, default=0)

    # Test
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    # About dataset
    if args.val_dir is None:
        args.val_dir = args.train_dir

    # About scheduler
    if args.scheduler == 'multi-step':
        assert all(s < args.epochs for s in args.schedule)
    else:
        assert args.schedule is None

    # About scaling
    assert args.scale > 0
    assert 0.0 < args.scale_min < args.scale_max < 1.0
    args.scale_range = args.scale_max - args.scale_min

    # About translation range
    args.trans_range = 2 * (1 - args.scale_min)
    args.trans_min = -args.trans_range / 2

    # About the number of iterations
    # We left it easy to explore with different coefficients for each iteration
    # Note that we will then normalize the coefficients so that their sum is 1
    assert args.n_iter > 0
    if args.n_iter_coef is None:
        args.n_iter_coef = [1] * args.n_iter
    assert len(args.n_iter_coef) == args.n_iter

    # About n_iter == 1
    if args.n_iter == 1:
        args.s = 1.0
        args.clip = 0.0
        args.alpha = 0.0
        args.ss_coef = 0.0
        args.ssl_coef = 0.0
        args.div_coef = 0.0

    # About glimpse classifier
    assert 0.0 <= args.alpha <= 1.0
    args.aux = args.alpha > 0.0 and args.n_iter >= 2

    # About loss to the glimpse-region size
    args.ss = args.ss_coef > 0.0

    # About self supervised spatial guidance
    assert 0.0 <= args.ssl_explore <= 1.0
    args.ssl = args.ssl_coef > 0.0

    # About diversity of glimpse-regions
    args.div = args.div_coef > 0.0 and args.n_iter >= 3

    # About exploration of exploding gradient problems
    assert 0.0 < args.s <= 1.0
    assert 0.0 <= args.clip <= 1.0
    if args.ge_final is None:
        args.ge_final = args.ge

    # About logs
    if args.tmp:
        args.log_dir = os.path.join('/tmp', args.log_dir)
    args.log_dir = os.path.join(
        args.log_dir, f'{args.dataset}{args.num_class}', f'{args.model}')

    args.world_size = len(list(filter(None, args.gpus.split(','))))
    args.batch_size //= args.world_size

    main(args)
