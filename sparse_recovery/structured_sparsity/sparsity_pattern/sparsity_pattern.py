class sparsity_pattern(object):
    def __init__(self, sig_dim=2):
        if sig_dim < 1:
            raise Exception('sparsity_pattern : Signal dimension cannot be less than 1. Quiting...')
        self.sig_dim = sig_dim


