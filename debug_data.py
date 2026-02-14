import polars as pl

DATA_PATH = "merged_training_data.parquet"

try:
    print(f"Loading {DATA_PATH}...")
    df = pl.read_parquet(DATA_PATH).head(10000)
    
    print("\nSchema:")
    print(df.schema)
    
    print("\nClick Column Stats:")
    if "click" in df.columns:
        print(f"Total Clicks: {df['click'].sum()}")
        print(f"Total Count: {len(df)}")
        print(f"First element: {df['click'][0]} (Type: {type(df['click'][0])})")
    else:
        print("'click' column missing!")

    print("\nConversion Column Stats:")
    if "conversion" in df.columns:
        print(f"Total Conversions: {df['conversion'].sum()}")
    else:
        print("'conversion' column missing!")

    print("\nPayprice Stats:")
    if "payprice" in df.columns:
        print(f"Min: {df['payprice'].min()}")
        print(f"Max: {df['payprice'].max()}")
        print(f"Mean: {df['payprice'].mean()}")
    else:
        print("'payprice' column missing!")
        
except Exception as e:
    print(f"Error: {e}")
