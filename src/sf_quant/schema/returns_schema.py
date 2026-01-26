import dataframely as dy

class SecurityRetSchema(dy.Schema):
    date = dy.Date(nullable=False)
    barrid = dy.String(nullable=False)
    _return = dy.Float64(nullable=False, alias="return")

class PortfolioRetSchema(dy.Schema):
    date = dy.Date(nullable=False)
    _return = dy.Float64(nullable=False, alias="return")

class MultiPortfolioRetSchema(dy.Schema):
    date = dy.Date(nullable=False)
    portfolio = dy.String(nullable=False)
    _return = dy.Float64(nullable=False, alias="return")

