import dataframely as dy

class ICSchema(dy.Schema):
    date = dy.Date(nullable=False)
    ic = dy.Float64(nullable=False)
    n = dy.Int64(nullable=False)
    ic_type = dy.String(nullable=False)