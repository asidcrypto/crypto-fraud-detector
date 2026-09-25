import requests
import pandas as pd

# Paste your Etherscan API Key between the quotes below
ETHERSCAN_API_KEY = J6QYE89MGTKJKFZBYSP3M1B4UM8A9R79RS

def get_wallet_features(address):
    url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={ETHERSCAN_API_KEY}"
    
    response = requests.get(url)
    data = response.json()
    
    if data['status'] != '1' or not data['result']:
        print("Error fetching data or no transactions found for this address.")
        return None
        
    txs = data['result']
    df = pd.DataFrame(txs)
    
    # Clean numeric fields
    df['timeStamp'] = df['timeStamp'].astype(float)
    df['value'] = df['value'].astype(float) / 1e18  # Convert Wei to ETH
    
    address_lower = address.lower()
    
    # Separate outgoing and incoming transactions
    sent_txs = df[df['from'].str.lower() == address_lower]
    recv_txs = df[df['to'].str.lower() == address_lower]
    
    # Calculate time differences in minutes
    if len(sent_txs) > 1:
        avg_min_sent = sent_txs['timeStamp'].diff().mean() / 60.0
    else:
        avg_min_sent = 0
        
    if len(recv_txs) > 1:
        avg_min_recv = recv_txs['timeStamp'].diff().mean() / 60.0
    else:
        avg_min_recv = 0
        
    time_diff_first_last = (df['timeStamp'].max() - df['timeStamp'].min()) / 60.0 if len(df) > 1 else 0
    
    # Build dictionary matching exact features used in model training
    features = {
        'Avg_min_between_sent_tnx': avg_min_sent,
        'Avg_min_between_received_tnx': avg_min_recv,
        'Time_Diff_between_first_and_last_(Mins)': time_diff_first_last,
        'Sent_tnx': len(sent_txs),
        'Received_Tnx': len(recv_txs),
        'Unique_Received_From_Addresses': recv_txs['from'].nunique() if len(recv_txs) > 0 else 0,
        'Unique_Sent_To_Addresses': sent_txs['to'].nunique() if len(sent_txs) > 0 else 0,
        'Total_Ether_Sent': sent_txs['value'].sum() if len(sent_txs) > 0 else 0,
        'Total_Ether_Received': recv_txs['value'].sum() if len(recv_txs) > 0 else 0
    }
    
    return pd.DataFrame([features])