class HParams():
    _skip_keys = ['to_dict']
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        if not hasattr(self, item):
            raise KeyError(f"Hyperparameter {item} has not been declared yet")
        setattr(self, item, value)

    def to_dict(self):
        res = {}
        for k in dir(self):
            if k.startswith('_') or k in self._skip_keys:
                continue
            res[k] =  self[k]
        return res