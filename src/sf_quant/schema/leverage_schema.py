import dataframely as dy

class LeverageSchema(dy.Schema):
    date = dy.Date(nullable=False)
    leverage = dy.Float64(nullable=False)