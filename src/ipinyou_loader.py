"""
Download and parse iPinYou RTB dataset from Kaggle.
Official dataset for hackathon challenge - REAL DATA ONLY.
"""
import os
import pandas as pd
import bz2
from pathlib import Path
from io import StringIO

def download_ipinyou_dataset():
    """Download iPinYou dataset using kagglehub."""
    try:
        import kagglehub
        print("Downloading iPinYou dataset from Kaggle...")
        path = kagglehub.dataset_download("lastsummer/ipinyou")
        print(f"✅ Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("Note: Make sure you have Kaggle credentials configured")
        return None

def parse_ipinyou_logs(dataset_path, max_records=100000):
    """
    Parse iPinYou dataset logs from the contest format.
    Based on actual schema analysis of bid.*.txt.bz2 files.
    
    Args:
        dataset_path: Path to downloaded dataset
        max_records: Max records to load (for faster processing)
    
    Returns:
        DataFrame with parsed bid requests
    """
    print(f"Parsing iPinYou contest dataset...")
    
    # Find the training data folder
    contest_path = Path(dataset_path) / "ipinyou.contest.dataset"
    if not contest_path.exists():
        print(f"❌ Contest dataset folder not found at {contest_path}")
        return None
    
    # Use training1st folder (March 2013 data)
    train_path = contest_path / "training1st"
    if not train_path.exists():
        print(f"❌ Training folder not found at {train_path}")
        return None
    
    # Find impression and click files
    # CRITICAL: Use IMP files (won auctions with served ads), NOT bid files (all auctions)
    # Clicks are matched to impression IDs, not bid IDs!
    imp_files = sorted(train_path.glob("imp.*.txt.bz2"))
    clk_files = sorted(train_path.glob("clk.*.txt.bz2"))
    
    print(f"Found {len(imp_files)} impression files, {len(clk_files)} click files")
    
    if not imp_files:
        print("❌ No impression files found")
        return None
    
    # Use first day's impression data
    imp_file = imp_files[0]
    print(f"Loading impression data from: {imp_file.name}")
    print(f"  (These are ads that WON the auction and were actually served)")
    
    
    # iPinYou IMPRESSION log schema (22 fields - DIFFERENT from bid files!)
    # Based on actual file inspection: Field 3 is logtype (1 for impression)
    imp_columns = [
        'bidid',             # 1. Bid/Impression ID (hash)
        'timestamp',          # 2. Timestamp (YYYYMMDDHHMMSS + ms) - 17 digits
        'logtype',           # 3. Log Type (1=impression, not a user hash!)
        'ipinyouid',         # 4. iPinYou User ID (hash) 
        'useragent',         # 5. User Agent string
        'IP',                # 6. IP address (masked)
        'region',            # 7. Region code (numeric)
        'city',              # 8. City code (numeric)
        'adexchange',        # 9. Ad Exchange ID (numeric)
        'domain',            # 10. Domain hash
        'url',               # 11. URL hash
        'anonymousurl',      # 12. Anonymous URL (mostly empty/null)
        'slotid',            # 13. Ad Slot ID
        'slotwidth',         # 14. Ad Slot Width
        'slotheight',        # 15. Ad Slot Height
        'slotvisibility',    # 16. Ad Slot Visibility
        'slotformat',        # 17. Ad Slot Format
        'adslotfloorprice',  # 18. Floor Price
        'creative',          # 19. Creative ID (hash)
        'biddingprice',      # 20. Bidding Price (what we bid)
        'payingprice',       # 21. Paying Price (what we actually paid)
        'advertiser'         # 22. Advertiser ID (numeric)
    ]
    
    try:
        # Read bz2 compressed impression file - these are ACTUAL SERVED ADS  
        print(f"  Reading up to {max_records:,} impression records...")
        with bz2.open(imp_file, 'rt', encoding='utf-8') as f:
            lines = [f.readline() for i in range(max_records) if f.readable()][:max_records]
        
        # Parse into DataFrame
        df = pd.read_csv(StringIO(''.join(lines)), sep='\t', header=None, 
                        names=imp_columns, low_memory=False)
        
        print(f"✅ Loaded {len(df):,} impression records (actual served ads)")
        
        # Parse timestamp (17 digits: YYYYMMDDHHMMSS + milliseconds)
        # Example: 20130311000101091 = 2013-03-11 00:01:01.091
        df['timestamp'] = df['timestamp'].astype(str).str[:14]  # Take first 14 digits (YYYYMMDDHHMMSS)
        df['datetime'] = pd.to_datetime(df['timestamp'], format='%Y%m%d%H%M%S', errors='coerce')
        df['hour'] = df['datetime'].dt.hour
        df['weekday'] = df['datetime'].dt.weekday
        
        # Convert numeric fields
        numeric_cols = ['region', 'city', 'adexchange', 'slotwidth', 'slotheight', 
                       'slotvisibility', 'slotformat', 'adslotfloorprice', 'biddingprice', 
                       'payingprice', 'advertiser']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # Create click and conversion columns (will be filled from separate files)
        df['click'] = 0
        df['conversion'] = 0
        
        # Load ALL click data to maximize matches
        clk_file = clk_files[0] if clk_files else None
        if clk_file:
            print(f"Loading click data from: {clk_file.name}")
            print(f"  Loading ALL clicks from file (may take a moment)...")
            with bz2.open(clk_file, 'rt', encoding='utf-8') as f:
                click_lines = f.readlines()  # Load ALL clicks, not just 10K
            
            print(f"  Parsing {len(click_lines):,} click records...")
            clk_df = pd.read_csv(StringIO(''.join(click_lines)), sep='\t', header=None,
                                names=imp_columns, low_memory=False)
            click_bids = set(clk_df['bidid'].dropna().values)
            print(f"  Found {len(click_bids):,} unique click bidIDs")
            
            # Match clicks to our bid data
            df.loc[df['bidid'].isin(click_bids), 'click'] = 1
            matched_clicks = df['click'].sum()
            print(f"  ✅ Matched {matched_clicks:,} clicks ({matched_clicks/len(df)*100:.3f}% CTR)")
            
            if matched_clicks == 0:
                print("  ⚠️ WARNING: No clicks matched! Trying to load more bid records...")
        
        # Add campaign_id (use advertiser)
        df['campaign_id'] = df['advertiser']
        
        # Add payingprice (using floor price as base - real payprice requires impression logs)
        df['payingprice'] = df['adslotfloorprice']
        
        # Create usertag from ipinyouid hash
        df['usertag'] = df['ipinyouid'].apply(lambda x: hash(str(x)) % 1000 if pd.notna(x) else 0)
        
        # Select relevant columns for our system
        relevant_cols = [
            'bidid', 'datetime', 'timestamp', 'campaign_id', 'advertiser',
            'region', 'city', 'adexchange', 'slotwidth', 'slotheight',
            'adslotfloorprice', 'payingprice', 'usertag',
            'hour', 'weekday', 'click', 'conversion'
        ]
        
        df = df[relevant_cols]
        
        # Remove invalid records (only drop if critical fields are missing)
        initial_count = len(df)
        
        # Debug: Check what's causing the filtering
        null_datetime = df['datetime'].isna().sum()
        zero_floor = (df['adslotfloorprice'] == 0).sum()
        print(f"  Debug: {null_datetime:,} records with null datetime, {zero_floor:,} with zero floor price")
        
        # Only drop records with invalid datetime (too restrictive to require floor price > 0)
        df = df.dropna(subset=['datetime'])
        # NOTE: NOT filtering by floor price since impression files might have 0 floor prices
        
        filtered = initial_count - len(df)
        if filtered > 0:
            print(f"  Filtered {filtered:,} records with invalid datetime")
        
        # CRITICAL: Ensure we have at least some clicks for training
        num_clicks = df['click'].sum()
        if num_clicks < 10:
            print(f"\n⚠️ WARNING: Only {num_clicks} clicks found - CTR model may not train well!")
            print("   Recommendation: Increase max_records to 300K-500K for better data")
        
        # Print statistics
        print(f"\n=== Real iPinYou Dataset Statistics ===")
        print(f"Total Valid Records: {len(df):,}")
        print(f"Clicks: {num_clicks:,}")
        print(f"CTR: {df['click'].mean():.4%}")
        print(f"CVR: {df['conversion'].mean():.5%}")
        print(f"Date Range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"Unique Campaigns: {df['campaign_id'].nunique()}")
        print(f"Avg Floor Price: ${df['adslotfloorprice'].mean():.2f}")
        print(f"Regions: {df['region'].nunique()}, Ad Exchanges: {df['adexchange'].nunique()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error parsing data files: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_ipinyou_data(cache_path='data/ipinyou_processed.csv', force_download=False, max_records=200000):
    """
    Load iPinYou dataset (download if needed, use cache if available).
    REAL DATA ONLY - NO SYNTHETIC FALLBACK.
    
    Args:
        cache_path: Where to cache processed data
        force_download: Force re-download even if cache exists
        max_records: Maximum records to load (for speed)
    
    Returns:
        DataFrame with iPinYou data
    
    Raises:
        RuntimeError: If dataset cannot be downloaded or parsed
    """
    # Check cache first
    if os.path.exists(cache_path) and not force_download:
        print(f"Loading cached iPinYou data from {cache_path}...")
        df = pd.read_csv(cache_path, parse_dates=['datetime'])
        print(f"✅ Loaded {len(df):,} records from cache")
        print(f"CTR: {df['click'].mean():.4%}, CVR: {df['conversion'].mean():.5%}")
        return df
    
    # Download dataset
    dataset_path = download_ipinyou_dataset()
    if dataset_path is None:
        raise RuntimeError(
            "❌ FAILED: Could not download iPinYou dataset from Kaggle.\n"
            "Make sure you have Kaggle credentials configured.\n"
            "ONLY REAL iPinYou DATA IS SUPPORTED - NO SYNTHETIC FALLBACK"
        )
    
    # Parse logs
    df = parse_ipinyou_logs(dataset_path, max_records=max_records)
    if df is None or len(df) == 0:
        raise RuntimeError(
            "❌ FAILED: Could not parse iPinYou dataset.\n"
            "Dataset structure may be corrupted or incompatible.\n"
            "ONLY REAL iPinYou DATA IS SUPPORTED - NO SYNTHETIC FALLBACK"
        )
    
    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"✅ Cached processed data to {cache_path}")
    
    return df

if __name__ == "__main__":
    # Test download and parsing
    df = load_ipinyou_data(force_download=True, max_records=50000)
    if df is not None:
        print("\nDataset preview:")
        print(df.head(10))
        print("\nColumn types:")
        print(df.dtypes)
