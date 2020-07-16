import torch.nn as nn


class Network(nn.Module):

    def forward(self, *input):
        raise NotImplementedError("Abstract Network method called. "
                                  "Concrete classes should override this method.")

    def loss(self, input, target):
        raise NotImplementedError("Abstract Network method called. "
                                  "Concrete classes should override this method.")

    def init_params(self):
        raise NotImplementedError("Abstract Network method called. "
                                  "Concrete classes should override this method.")

    def log_parameters(self, logger, global_step):
        pass

    def log_validation(self, logger, global_step):
        pass
