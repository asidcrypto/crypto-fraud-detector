import requests
import pandas as pd

ETHERSCAN_API_KEY = "J6QYE89MGTKJKFZBYSP3M1B4UM8A9R79RS"

def get_wallet_features(address):
    # Etherscan API V2 unified endpoint for Ethereum Mainnet (chainid=1)
    url = f"https://api.etherscan.io/v2/api?chainid=1&module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={ETHERSCAN_API_KEY}"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
    except Exception as e:
        print(f"Request failed: {e}")
        return None
    
    # Check if result contains valid transaction list
    result = data.get('result')
    if not isinstance(result, list) or len(result) == 0:
        return None
        
    df = pd.DataFrame(result)
    
    if df.empty or 'timeStamp' not in df.columns:
        return None

    # Format values safely
    df['timeStamp'] = pd.to_numeric(df['timeStamp'], errors='coerce').fillna(0)
    df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0) / 1e18  # Convert Wei to ETH
    
    address_lower = address.lower()
    
    # Separate sent vs received transactions
    sent_txs = df[df['from'].astype(str).str.lower() == address_lower]
    recv_txs = df[df['to'].astype(str).str.lower() == address_lower]
    
    # Calculate time intervals in minutes
    avg_min_sent = sent_txs['timeStamp'].diff().mean() / 60.0 if len(sent_txs) > 1 else 0.0
    avg_min_recv = recv_txs['timeStamp'].diff().mean() / 60.0 if len(recv_txs) > 1 else 0.0
    time_diff_first_last = (df['timeStamp'].max() - df['timeStamp'].min()) / 60.0 if len(df) > 1 else 0.0
    
    # Construct exact feature map expected by fraud_model.pkl
    features = {
        'Avg_min_between_sent_tnx': float(avg_min_sent) if not pd.isna(avg_min_sent) else 0.0,
        'Avg_min_between_received_tnx': float(avg_min_recv) if not pd.isna(avg_min_recv) else 0.0,
        'Time_Diff_between_first_and_last_(Mins)': float(time_diff_first_last) if not pd.isna(time_diff_first_last) else 0.0,
        'Sent_tnx': float(len(sent_txs)),
        'Received_Tnx': float(len(recv_txs)),
        'Unique_Received_From_Addresses': float(recv_txs['from'].nunique()) if len(recv_txs) > 0 else 0.0,
        'Unique_Sent_To_Addresses': float(sent_txs['to'].nunique()) if len(sent_txs) > 0 else 0.0,
        'Total_Ether_Sent': float(sent_txs['value'].sum()) if len(sent_txs) > 0 else 0.0,
        'Total_Ether_Received': float(recv_txs['value'].sum()) if len(recv_txs) > 0 else 0.0
    }
    
    return pd.DataFrame([features])