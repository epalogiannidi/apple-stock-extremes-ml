class ModelTrainReturnObject:
    """Defines the structure of the trained model"""

    def __init__(self, model=None, epoch=0, desc="", params={}):
        self.model = model
        self.epoch = epoch
        self.desc = desc
        self.params = params


class ModelMetricsObject:
    """Defines the evaluation metrics"""

    def __init__(self, cl_report, conf_matrix, roc_auc):
        self.cl_report = cl_report
        self.conf_matrix = conf_matrix
        self.roc_auc = roc_auc
