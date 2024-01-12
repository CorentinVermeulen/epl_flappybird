class AbstractAgent:

    def set_hyperparameters(self, hyperparameters):
        raise NotImplementedError
    def train(self):
        raise NotImplementedError

    def retrain(self,**kwargs):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
