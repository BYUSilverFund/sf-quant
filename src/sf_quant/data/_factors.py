factors = sorted(
    [
        "USSLOWL_BETA",
        "USSLOWL_COUNTRY",
        "USSLOWL_DIVYILD",
        "USSLOWL_EARNQLTY",
        "USSLOWL_EARNYILD",
        "USSLOWL_GROWTH",
        "USSLOWL_LEVERAGE",
        "USSLOWL_LIQUIDTY",
        "USSLOWL_LTREVRSL",
        "USSLOWL_MGMTQLTY",
        "USSLOWL_MIDCAP",
        "USSLOWL_MOMENTUM",
        "USSLOWL_PROFIT",
        "USSLOWL_PROSPECT",
        "USSLOWL_SIZE",
        "USSLOWL_VALUE",
        # "USSLOWL_NETRET",
        "USSLOWL_AERODEF",
        "USSLOWL_AIRLINES",
        "USSLOWL_ALUMSTEL",
        "USSLOWL_APPAREL",
        "USSLOWL_AUTO",
        "USSLOWL_BANKS",
        "USSLOWL_BEVTOB",
        "USSLOWL_BIOLIFE",
        "USSLOWL_BLDGPROD",
        "USSLOWL_CHEM",
        "USSLOWL_CNSTENG",
        "USSLOWL_CNSTMACH",
        "USSLOWL_CNSTMATL",
        "USSLOWL_COMMEQP",
        "USSLOWL_COMPELEC",
        "USSLOWL_COMSVCS",
        "USSLOWL_CONGLOM",
        "USSLOWL_CONTAINR",
        "USSLOWL_DISTRIB",
        "USSLOWL_DIVFIN",
        "USSLOWL_ELECEQP",
        "USSLOWL_ELECUTIL",
        "USSLOWL_FOODPROD",
        "USSLOWL_FOODRET",
        "USSLOWL_GASUTIL",
        "USSLOWL_HLTHEQP",
        "USSLOWL_HLTHSVCS",
        "USSLOWL_HOMEBLDG",
        "USSLOWL_HOUSEDUR",
        "USSLOWL_INDMACH",
        "USSLOWL_INSURNCE",
        "USSLOWL_INTERNET",
        "USSLOWL_LEISPROD",
        "USSLOWL_LEISSVCS",
        "USSLOWL_LIFEINS",
        "USSLOWL_MEDIA",
        "USSLOWL_MGDHLTH",
        "USSLOWL_MULTUTIL",
        "USSLOWL_OILGSCON",
        "USSLOWL_OILGSDRL",
        "USSLOWL_OILGSEQP",
        "USSLOWL_OILGSEXP",
        "USSLOWL_PAPER",
        "USSLOWL_PHARMA",
        "USSLOWL_PRECMTLS",
        "USSLOWL_PSNLPROD",
        "USSLOWL_REALEST",
        "USSLOWL_RESTAUR",
        "USSLOWL_RESVOL",
        "USSLOWL_ROADRAIL",
        "USSLOWL_SEMICOND",
        "USSLOWL_SEMIEQP",
        "USSLOWL_SOFTWARE",
        "USSLOWL_SPLTYRET",
        "USSLOWL_SPTYCHEM",
        "USSLOWL_SPTYSTOR",
        "USSLOWL_TELECOM",
        "USSLOWL_TRADECO",
        "USSLOWL_TRANSPRT",
        "USSLOWL_WIRELESS",
    ]
)

style_factors = ['USSLOWL_BETA','USSLOWL_DIVYILD','USSLOWL_EARNQLTY','USSLOWL_EARNYILD','USSLOWL_GROWTH',
                 'USSLOWL_LEVERAGE','USSLOWL_LTREVRSL','USSLOWL_MGMTQLTY','USSLOWL_MIDCAP','USSLOWL_MOMENTUM',
                 'USSLOWL_NETRET','USSLOWL_PROFIT','USSLOWL_PROSPECT','USSLOWL_RESVOL','USSLOWL_SIZE','USSLOWL_VALUE']

sector_factors = sorted(
    [
        'USSLOWL_AERODEF', #(Aerospace & Defense)
        'USSLOWL_AIRLINES',
        'USSLOWL_ALUMSTEL', #(Aluminum & Steel)
        'USSLOWL_APPAREL',
        'USSLOWL_AUTO',
        'USSLOWL_BANKS',
        'USSLOWL_BEVTOB', #(Beverages & Tobacco)
        'USSLOWL_BIOLIFE', #(Biotechnology/Life Sciences)
        'USSLOWL_BLDGPROD', #(Building Products)
        'USSLOWL_CHEM', #(Chemicals)
        'USSLOWL_CNSTENG', #(Construction & Engineering)
        'USSLOWL_CNSTMACH', #(Construction Machinery)
        'USSLOWL_CNSTMATL', #(Construction Materials)
        'USSLOWL_COMMEQP', #(Commercial Equipment)
        'USSLOWL_COMPELEC', #(Computer Electronics)
        'USSLOWL_COMSVCS', #(Commercial Services)
        'USSLOWL_CONGLOM', #(Conglomerates)
        'USSLOWL_CONTAINR', #(Containers & Packaging)
        'USSLOWL_DISTRIB', #(Distributors)
        'USSLOWL_DIVFIN', #(Diversified Financials)
        'USSLOWL_ELECEQP', #(Electrical Equipment)
        'USSLOWL_ELECUTIL', #(Electric Utilities)
        'USSLOWL_FOODPROD',
        'USSLOWL_FOODRET',
        'USSLOWL_GASUTIL',
        'USSLOWL_HLTHEQP', #(Health Equipment)
        'USSLOWL_HLTHSVCS', #(Health Services)
        'USSLOWL_HOMEBLDG',
        'USSLOWL_HOUSEDUR', #(Household Durables)
        'USSLOWL_INDMACH', #(Industrial Machinery)
        'USSLOWL_INSURNCE',
        'USSLOWL_INTERNET',
        'USSLOWL_LEISPROD', #(Leisure Products)
        'USSLOWL_LEISSVCS', #(Leisure Services)
        'USSLOWL_LIFEINS',
        'USSLOWL_MEDIA',
        'USSLOWL_MGDHLTH', #(Managed Health Care)
        'USSLOWL_MULTUTIL', #(Multi-Utilities)
        'USSLOWL_OILGSCON', #(Oil & Gas Consumables)
        'USSLOWL_OILGSDRL', #(Oil & Gas Drilling)
        'USSLOWL_OILGSEQP', #(Oil & Gas Equipment)
        'USSLOWL_OILGSEXP', #(Oil & Gas Exploration)
        'USSLOWL_PAPER',
        'USSLOWL_PHARMA',
        'USSLOWL_PRECMTLS', #(Precious Metals)
        'USSLOWL_PSNLPROD', #(Personal Products)
        'USSLOWL_REALEST', #(Real Estate)
        'USSLOWL_RESTAUR',
        'USSLOWL_RESVOL', #(likely Resorts & Leisure)
        'USSLOWL_ROADRAIL',
        'USSLOWL_SEMICOND',
        'USSLOWL_SEMIEQP', #(Semiconductor Equipment)
        'USSLOWL_SOFTWARE',
        'USSLOWL_SPLTYRET', #(Specialty Retail)
        'USSLOWL_SPTYCHEM', #(Specialty Chemicals)
        'USSLOWL_SPTYSTOR', #(Specialty Stores)
        'USSLOWL_TELECOM',
        'USSLOWL_TRADECO',
        'USSLOWL_TRANSPRT',
        'USSLOWL_WIRELESS',
    ]
)