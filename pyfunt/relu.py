from threshold import Threshold


class ReLU(Threshold):
    def __init__(self, ip=False):
        super(ReLU, self).__init__(0, 0, ip)
