class Track():
    _id = None
    _box = None
    _score = None
    _class = None

    def __init__(self, id, box, score, clazz):
        self._id = id
        self._box = box
        self._score = score
        self._class = clazz

    @property
    def center(self):
        return (self.center_x, self.center_y)
    
    @property
    def center_x(self):
        return int((self._box[0] + self._box[2])/2)
    
    @property
    def center_y(self):
        return int((self._box[1] + self._box[3])/2)
      