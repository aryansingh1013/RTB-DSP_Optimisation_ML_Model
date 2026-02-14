import polars as pl
try:
    df = pl.read_parquet("d:/4sem/project/merged_training_data.parquet")
    print(df.columns)
    print(df.head(1))
except Exception as e:
    print(e)
