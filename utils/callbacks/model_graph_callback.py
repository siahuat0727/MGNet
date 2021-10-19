from pytorch_lightning.callbacks.base import Callback


class ModelAddGraph(Callback):
    """
    Add model graph
    """

    def on_validation_start(self, trainer, pl_module):
        # TODO to device?
        images = next(iter(pl_module.test_dataloader()))[
            0][:1].to(pl_module.device)
        trainer.logger.experiment.add_graph(pl_module.model, images)
