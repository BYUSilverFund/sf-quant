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


class EnvNotConfiguredError(Exception):
    pass


def get_base_path(table_name: str) -> str:
    missing = [key for key in ("root", "database") if not _config[key]]
    if missing:
        missing_vars = ", ".join(var.upper() for var in missing)
        raise EnvNotConfiguredError(
            f"Environment not configured: {missing_vars} not set.\n"
            f"Configure with:\n"
            f"  import sf_quant.data as sfd\n"
            f"  sfd.env(root='/path/to/root', database='your_database')\n"
            f"Or set ROOT and DATABASE environment variables in a .env file."
        )
    return f"{_config['root']}/groups/grp_quant/database/{_config['database']}/{table_name}"
