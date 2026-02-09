import os
import dotenv
import polars as pl

dotenv.load_dotenv(override=True)


class Table:
    def __init__(self, name: str, base_path: str) -> None:
        self._name = name
        self._base_path = base_path

    def _file_path(self, year: int | None = None) -> str:
        if year is None:
            return f"{self._base_path}/{self._name}_*.parquet"
        else:
            return f"{self._base_path}/{self._name}_{year}.parquet"

    def scan(self, year: int | None = None) -> pl.LazyFrame:
        return pl.scan_parquet(self._file_path(year))

    def read(self, year: int | None = None) -> pl.DataFrame:
        return pl.read_parquet(self._file_path(year))

    def columns(self) -> pl.DataFrame:
        pl.Config.set_tbl_rows(-1)
        schema = self.scan().collect_schema()
        df_str = str(
            pl.DataFrame(
                {
                    "column": list(schema.keys()),
                    "dtype": [str(t) for t in schema.values()],
                }
            )
        )
        pl.Config.set_tbl_rows(10)
        return df_str


class FamaFrenchTable(Table):
    def _file_path(self, year: int | None = None) -> str:
        if year is None:
            return f"{self._base_path}/{self._name}*.parquet"
        else:
            return f"{self._base_path}/{self._name}{year}.parquet"


assets_table = Table("assets", os.getenv("ASSETS_TABLE"))
crsp_daily_table = Table("crsp_daily", os.getenv("CRSP_DAILY_TABLE"))
crsp_v2_daily_table = Table("crsp_v2_daily", os.getenv("CRSP_V2_DAILY_TABLE"))
crsp_monthly_table = Table("crsp_monthly", os.getenv("CRSP_MONTHLY_TABLE"))
crsp_v2_monthly_table = Table("crsp_v2_monthly", os.getenv("CRSP_V2_MONTHLY_TABLE"))
crsp_events_table = Table("crsp_events", os.getenv("CRSP_EVENTS_TABLE"))
exposures_table = Table("exposures", os.getenv("EXPOSURES_TABLE"))
covariances_table = Table("covariances", os.getenv("COVARIANCES_TABLE"))
factors_table = Table("factors", os.getenv("FACTORS_TABLE"))
ff_table = FamaFrenchTable("ff5_factors", os.getenv("FF_TABLE"))
