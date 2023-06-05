from lightning.pytorch.callbacks import Callback
import lightning.pytorch as pl
import pandas as pd


class LossSaver(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not trainer.sanity_checking:
            metrics = {i: trainer.callback_metrics[i].item() for i in trainer.callback_metrics}
            self.metrics.append(metrics)

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        df = pd.DataFrame(self.metrics)
        df.to_excel(f'{trainer.model.__class__.__name__}.xlsx')
