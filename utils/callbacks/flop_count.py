from pytorch_lightning.callbacks.base import Callback
# from torchsummary import summary
from fvcore.nn.flop_count import flop_count


class FlopCount(Callback):
    """
    Get model flop
    """

    def on_test_end(self, trainer, pl_module):
        images = next(iter(trainer.datamodule.val_dataloader()))[
            0][:1].to(pl_module.device)

        pl_module.model.eval()
        gflop_dict, _ = flop_count(pl_module.model, (images, ))
        gflops = sum(gflop_dict.values())
        print(_)
        print(gflop_dict)
        print(f'{gflops=}')
        print(f'{pl_module.model.training=}')
        return gflops
