import dataframely as dy

class SecurityRetSchema(dy.Schema):
    date = dy.Date(nullable=False)
    barrid = dy.String(nullable=False)
    fwd_return = dy.Float64(nullable=False, alias="return")

class PortfolioRetSchema(dy.Schema):
    date = dy.Date(nullable=False)
    fwd_return = dy.Float64(nullable=False, alias="return")