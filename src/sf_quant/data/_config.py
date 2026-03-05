import os
import dotenv

dotenv.load_dotenv(override=True)

_config = {
    "root": os.environ.get("ROOT", ""),
    "database": os.environ.get("DATABASE", ""),
}


def env(root: str | None = None, database: str | None = None) -> None:
    if root is not None:
        _config["root"] = root
    if database is not None:
        _config["database"] = database


def get_base_path(table_name: str) -> str:
    return f"{_config['root']}/groups/grp_quant/database/{_config['database']}/{table_name}"
