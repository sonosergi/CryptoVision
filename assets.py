class Asset:
    def __init__(self, fecha, apertura, máximo, mínimo, var):
        self.fecha = fecha
        self.apertura = apertura
        self.maximo = máximo
        self.mínimo = mínimo
        self. var = var

class Index(Asset):
    def __init__(self, fecha, apertura, máximo, mínimo, var):
        super().__init__(fecha, apertura, máximo, mínimo, var)

