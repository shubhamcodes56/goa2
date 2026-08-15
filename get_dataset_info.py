import duckdb
HF_TOKEN = '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'

try:
    con = duckdb.connect()
    con.execute(f"INSTALL httpfs;")
    con.execute(f"LOAD httpfs;")
    # Set HF token
    con.execute(f"SET s3_region='us-east-1';")
    print("Checking row count...")
    res = con.execute(f"SELECT count(*) FROM 'hf://datasets/ai4bharat/MSMARCO-XI/train/hintrain.parquet'").fetchone()
    print(f"Total rows in hintrain.parquet: {res[0]}")
except Exception as e:
    print(e)
