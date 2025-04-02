from apple_stock_extremes_ml.models.utils.early_stopper import EarlyStopper


def test_early_stopper_initialization():
    early_stopper = EarlyStopper()
    assert isinstance(early_stopper, EarlyStopper)
    assert early_stopper.patience == 5


def test_early_stopper(val_loss):
    early_stopper = EarlyStopper()
    for i in range(0, 10):
        res = early_stopper(i * val_loss)
