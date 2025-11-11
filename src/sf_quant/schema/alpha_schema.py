import dataframely as dy

class AlphaSchema(dy.Schema):
    date = dy.Date(nullable=False)
    barrid = dy.String(nullable=False)
    alpha = dy.Float64(nullable=False)