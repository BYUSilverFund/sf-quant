import dataframely as dy

class PortfolioSchema(dy.Schema):
    date = dy.Date(nullable=False)
    barrid = dy.String(nullable=False)
    weight = dy.Float64(nullable=False)