import datetime as dt
import pandas as pd

from webull.data.common.category import Category
from webull.data.common.timespan import Timespan
from webull.core.client import ApiClient
from webull.data.data_client import DataClient

from src.utils import config

def load_data(symbols: list, start_date: str, end_date: str) -> pd.DataFrame:
    # 1. Initialize API clients once outside the loop for efficiency
    api_client = ApiClient(config.get('webull_api.api_key'), config.get('webull_api.app_secret'), "th")
    api_client.add_endpoint("th", "api.webull.co.th")
    data_client = DataClient(api_client)

    # 2. Parse date windows and calculate optimal bars to request
    start_dt = dt.datetime.strptime(start_date, "%Y-%m-%d")
    end_date_parsed = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    today = dt.datetime.today()
    
    calendar_days = (today - start_dt).days
    estimated_trading_days = int(calendar_days * 0.7)
    fetch_count = min(estimated_trading_days + 30, 1200)

    all_ticker_dfs = []
    
    # Define a safe chunk size strictly less than 20
    CHUNK_SIZE = config.get('chunk_size')
    
    # 3. Iterate through symbols in batches
    for i in range(0, len(symbols), CHUNK_SIZE):
        chunk_symbols = symbols[i:i + CHUNK_SIZE]
        print(f"Batch Processing: Requesting data ({fetch_count} bars) for {chunk_symbols}...")
        
        try:
            res = data_client.market_data.get_batch_history_bar(
                chunk_symbols, 
                Category.US_STOCK.name,
                Timespan.D.name, 
                fetch_count
            )
            
            if res.status_code != 200:
                print(f"Skipping Batch: HTTP {res.status_code} encountered for {chunk_symbols}")
                continue
                
            response_data = res.json()
            if not response_data or "result" not in response_data:
                print(f"Skipping Batch: No result array found for {chunk_symbols}")
                continue

            ticker_list = response_data["result"]

            # 4. Parse individual ticker payloads inside the current batch
            for item in ticker_list:
                ticker = item.get("symbol")
                bars = item.get("result", [])
                
                if not bars:
                    print(f"Skipping {ticker}: No historical bars found.")
                    continue
                
                temp_df = pd.DataFrame(bars)
                temp_df['symbol'] = ticker  
                temp_df['date'] = pd.to_datetime(temp_df['time']).dt.date
                
                numeric_cols = ['open', 'close', 'high', 'low']
                for col in numeric_cols:
                    if col in temp_df.columns:
                        temp_df[col] = pd.to_numeric(temp_df[col])
                        
                if 'volume' in temp_df.columns:
                    temp_df['volume'] = pd.to_numeric(temp_df['volume']).astype(int)
                    
                if 'trading_session' in temp_df.columns:
                    temp_df['trading_session'] = temp_df['trading_session'].replace("", "RTH")

                # Filter down to the user's specific requested training interval
                filtered_temp_df = temp_df[(temp_df['date'] >= start_dt.date()) & (temp_df['date'] <= end_date_parsed)]
                all_ticker_dfs.append(filtered_temp_df)
                
        except Exception as e:
            print(f"Error encountered while executing batch {chunk_symbols}: {e}")
            continue

    # 5. Consolidate and format all collected historical records
    if all_ticker_dfs:
        master_df = pd.concat(all_ticker_dfs, ignore_index=True)
        
        preferred_order = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'trading_session']
        existing_cols = [col for col in preferred_order if col in master_df.columns]
        master_df = master_df[existing_cols]
        
        # Order chronologically and alphabetically by stock ticker
        master_df = master_df.sort_values(by=['date', 'symbol']).reset_index(drop=True)
        print(f"Data loading complete. Total matrix size generated: {master_df.shape}")
        return master_df
    else:
        print("No matching records found across any processed batches within your date range.")
        return pd.DataFrame()