from threshold import Threshold


class ReLU(Threshold):
    def __init__(self, p):
        super(ReLU, self).__init__(0, 0, p)
