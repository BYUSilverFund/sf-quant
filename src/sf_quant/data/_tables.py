import polars as pl

from ._config import get_base_path


class Table:
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def _base_path(self) -> str:
        return get_base_path(self._name)

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


assets_table = Table("assets")
crsp_daily_table = Table("crsp_daily")
crsp_v2_daily_table = Table("crsp_v2_daily")
crsp_monthly_table = Table("crsp_monthly")
crsp_v2_monthly_table = Table("crsp_v2_monthly")
crsp_events_table = Table("crsp_events")
exposures_table = Table("exposures")
covariances_table = Table("covariances")
factors_table = Table("factors")
ff_table = FamaFrenchTable("ff5_factors")
