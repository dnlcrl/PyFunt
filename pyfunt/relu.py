from threshold import Threshold


class ReLU(Threshold):
    def __init__(self, p):
        super(Relu, self).__init__(self, 0, 0, p)
