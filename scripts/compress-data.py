from pyarrow import json, parquet
import os

for file in os.listdir("data/json"):
    print(file)
    df = json.read_json(f"data/json/{file}")
    parquet.write_table(
        df, f"data/compressed/{file.split('.')[0]}.gz", compression="gzip"
    )
