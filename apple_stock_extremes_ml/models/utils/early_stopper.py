from apple_stock_extremes_ml import logger


class EarlyStopper:
    """
    Applies early stopping when training a neural network based on the validation loss and a patience variable.
    If the validation loss has not been improved for #patience epochs, the training stops.

    """

    def __init__(self, patience: int = 5):
        self.patience = patience
        self.counter = 0
        self.latest_validation_loss = float("inf")

    def __call__(self, validation_loss: float):

        if validation_loss >= self.latest_validation_loss - 1e-3:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(
                    f"Early stopping activated because the validation loss hasn't been improved "
                    f"for the last {self.patience} epochs."
                )
                return True
        else:
            self.counter = 0
        self.latest_validation_loss = validation_loss
        return False
