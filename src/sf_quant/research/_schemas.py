import dataframely as dy

class SignalSchema(dy.Schema):
    date = dy.Date(nullable=False)
    barrid = dy.String(nullable=False)
    fwd_return = dy.String(nullable=False)
    signal = dy.Float64(nullable=False)