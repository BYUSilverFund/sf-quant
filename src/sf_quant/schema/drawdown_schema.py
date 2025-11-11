import dataframely as dy

class DrawdownSchema(dy.Schema):
    date = dy.Date(nullable=False)
    drawdown = dy.Float64(nullable=False)
