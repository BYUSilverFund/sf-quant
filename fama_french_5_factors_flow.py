import urllib.request
import zipfile
from pathlib import Path
import polars as pl
import io


def fama_french_5_factors_flow() -> None:
    # URL for the Fama-French 3 factors (monthly data)
    ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"

    # Download the zip file
    urllib.request.urlretrieve(ff_url, "fama_french.zip")

    # Extract the CSV file
    with zipfile.ZipFile("fama_french.zip", "r") as zip_file:
        zip_file.extractall()

    # Read the CSV file with polars
    # Skip first 4 rows (header info) and read only the monthly data
    # The file has annual data after the monthly data, which we'll exclude
    with open("F-F_Research_Data_5_Factors_2x3_daily.csv", "r") as f:
        lines = f.readlines()

    # Find where monthly data ends (when we hit empty line or "Annual" section)
    monthly_end_idx = None
    for i, line in enumerate(lines[4:], start=4):  # Start after header rows
        if line.strip() == "" or "Annual" in line:
            print(line)
            monthly_end_idx = i
            break

    # Read only the monthly data section
    csv_data = "".join(lines[3:monthly_end_idx])

    # Parse with polars
    data = pl.read_csv(
        io.StringIO(csv_data),
        has_header=True,
        new_columns=["date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"],
    ).with_columns(
        [
            # Convert date format (YYYYMM) to actual date (end of month)
            pl.col("date").cast(pl.String).str.strptime(pl.Date, "%Y%m%d"),
            # Convert from percentage to decimal
            pl.col("mkt_rf").str.strip_chars().cast(pl.Float64).truediv(100),
            pl.col("smb").str.strip_chars().cast(pl.Float64).truediv(100),
            pl.col("hml").str.strip_chars().cast(pl.Float64).truediv(100),
            pl.col("rmw").str.strip_chars().cast(pl.Float64).truediv(100),
            pl.col("cma").str.strip_chars().cast(pl.Float64).truediv(100),
            pl.col("rf").str.strip_chars().cast(pl.Float64).truediv(100),
        ]
    )

    # Create output directory
    file_path = Path("data/fama_french/ff5_factors.parquet")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to parquet
    data.write_parquet(file_path)

    # Clean up downloaded files
    Path("fama_french.zip").unlink(missing_ok=True)
    Path("F-F_Research_Data_5_Factors_2x3_daily.csv").unlink(missing_ok=True)


if __name__ == "__main__":
    fama_french_5_factors_flow()
