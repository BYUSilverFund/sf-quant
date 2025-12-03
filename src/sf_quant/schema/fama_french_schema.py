import dataframely as dy

class FamaFrenchSchema(dy.Schema):
    date = dy.Date(nullable=False)
    mkt_rf = dy.Float64(nullable=False)
    smb = dy.Float64(nullable=False)
    hml = dy.Float64(nullable=False)
    rmw = dy.Float64(nullable=False)
    cma = dy.Float64(nullable=False)
    rf = dy.Float64(nullable=False)
