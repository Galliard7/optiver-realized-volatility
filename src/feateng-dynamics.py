import numpy as np
import pandas as pd
from scipy import stats

import concurrent.futures

import seaborn as sns

pd.set_option('max_rows', 300)
pd.set_option('max_columns', 300)

from tqdm.notebook import tqdm
import gc


def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff()

def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))



def additional_feats(df, suffix = ""):
    df[f'wap1_range{suffix}'] = df[f'wap1_max{suffix}'] - df[f'wap1_min{suffix}']
    df[f'wap2_range{suffix}'] = df[f'wap2_max{suffix}'] - df[f'wap2_min{suffix}']
    df[f'wap3_range{suffix}'] = df[f'wap3_max{suffix}'] - df[f'wap3_min{suffix}']
    df[f'wap4_range{suffix}'] = df[f'wap4_max{suffix}'] - df[f'wap4_min{suffix}']
    df[f'wapbal_range{suffix}'] = df[f'wapbal_max{suffix}'] - df[f'wapbal_min{suffix}']

    df[f'realvol_diff1{suffix}'] = df[f'real_vol1{suffix}'] - df[f'real_vol2{suffix}']
    df[f'realvol_diff2{suffix}'] = df[f'real_vol2{suffix}'] - df[f'real_vol3{suffix}']
    df[f'realvol_diff3{suffix}'] = df[f'real_vol3{suffix}'] - df[f'real_vol4{suffix}']
    df[f'realvol_diff_max{suffix}'] = df[f'real_vol1{suffix}'] - df[f'real_vol4{suffix}']

    df[f'spread_range{suffix}'] = df[f'spread_max{suffix}'] - df[f'spread_min{suffix}']
    df[f'spread2_range{suffix}'] = df[f'spread2_max{suffix}'] - df[f'spread2_min{suffix}']
    df[f'price_spread_range{suffix}'] = df[f'price_spread_max{suffix}'] - df[f'price_spread_min{suffix}']
    df[f'price_spread2_range{suffix}'] = df[f'price_spread2_max{suffix}'] - df[f'price_spread2_min{suffix}']
    df[f'bid_spread_range{suffix}'] = df[f'bid_spread_max{suffix}'] - df[f'bid_spread_min{suffix}']
    df[f'ask_spread_range{suffix}'] = df[f'ask_spread_max{suffix}'] - df[f'ask_spread_min{suffix}']
    df[f'bid_ask_spread_range{suffix}'] = df[f'bid_ask_spread_max{suffix}'] - df[f'bid_ask_spread_min{suffix}']
    df[f'total_volume_range{suffix}'] = df[f'total_volume_max{suffix}'] - df[f'total_volume_min{suffix}']
    df[f'volume_imbalance_range{suffix}'] = df[f'volume_imbalance_max{suffix}'] - df[f'volume_imbalance_min{suffix}']
    df[f'spread2_range{suffix}'] = df[f'spread2_max{suffix}'] - df[f'spread2_min{suffix}']

    df[f'bid_ask_diff_range{suffix}'] = df[f'bid_ask_diff_max{suffix}'] - df[f'bid_ask_diff_min{suffix}']
    df[f'bid_ask_diff2_range{suffix}'] = df[f'bid_ask_diff2_max{suffix}'] - df[f'bid_ask_diff2_min{suffix}']

    df[f'time_num_seconds_norm{suffix}'] = df[f'time_num_seconds{suffix}'] / 600


    df[f'time_dynamic_static_ratio{suffix}'] = df[f'time_OB_total_dynamic_seconds{suffix}'] / df[f'time_OB_static_seconds_sum{suffix}']

    df[f'OB_bid_price1_dollar_change_range{suffix}'] = df[f'OB_bid_price1_dollar_change_max{suffix}'] - df[f'OB_bid_price1_dollar_change_min{suffix}']
    df[f'OB_bid_price1_same_change_ratio{suffix}'] = df[f'OB_bid_price1_same_sum{suffix}'] / df[f'OB_bid_price1_change_sum{suffix}']
    df[f'OB_bid_price1_up_down_ratio{suffix}'] = df[f'OB_bid_price1_up_sum{suffix}'] / df[f'OB_bid_price1_down_sum{suffix}']

    df[f'OB_bid_price2_dollar_change_range{suffix}'] = df[f'OB_bid_price2_dollar_change_max{suffix}'] - df[f'OB_bid_price2_dollar_change_min{suffix}']
    df[f'OB_bid_price2_same_change_ratio{suffix}'] = df[f'OB_bid_price2_same_sum{suffix}'] / df[f'OB_bid_price2_change_sum{suffix}']
    df[f'OB_bid_price2_up_down_ratio{suffix}'] = df[f'OB_bid_price2_up_sum{suffix}'] / df[f'OB_bid_price2_down_sum{suffix}']

    df[f'OB_ask_price1_dollar_change_range{suffix}'] = df[f'OB_ask_price1_dollar_change_max{suffix}'] - df[f'OB_ask_price1_dollar_change_min{suffix}']
    df[f'OB_ask_price1_same_change_ratio{suffix}'] = df[f'OB_ask_price1_same_sum{suffix}'] / df[f'OB_ask_price1_change_sum{suffix}']
    df[f'OB_ask_price1_up_down_ratio{suffix}'] = df[f'OB_ask_price1_up_sum{suffix}'] / df[f'OB_ask_price1_down_sum{suffix}']

    df[f'OB_ask_price2_dollar_change_range{suffix}'] = df[f'OB_ask_price2_dollar_change_max{suffix}'] - df[f'OB_ask_price2_dollar_change_min{suffix}']
    df[f'OB_ask_price2_same_change_ratio{suffix}'] = df[f'OB_ask_price2_same_sum{suffix}'] / df[f'OB_ask_price2_change_sum{suffix}']
    df[f'OB_ask_price2_up_down_ratio{suffix}'] = df[f'OB_ask_price2_up_sum{suffix}'] / df[f'OB_ask_price2_down_sum{suffix}']

    df[f'OB_wap1_dollar_change_range{suffix}'] = df[f'OB_wap1_dollar_change_max{suffix}'] - df[f'OB_wap1_dollar_change_min{suffix}']
    df[f'OB_wap1_same_change_ratio{suffix}'] = df[f'OB_wap1_same_sum{suffix}'] / df[f'OB_wap1_change_sum{suffix}']
    df[f'OB_wap1_up_down_ratio{suffix}'] = df[f'OB_wap1_up_sum{suffix}'] / df[f'OB_wap1_down_sum{suffix}']

    df[f'OB_wap2_dollar_change_range{suffix}'] = df[f'OB_wap2_dollar_change_max{suffix}'] - df[f'OB_wap2_dollar_change_min{suffix}']
    df[f'OB_wap2_same_change_ratio{suffix}'] = df[f'OB_wap2_same_sum{suffix}'] / df[f'OB_wap2_change_sum{suffix}']
    df[f'OB_wap2_up_down_ratio{suffix}'] = df[f'OB_wap2_up_sum{suffix}'] / df[f'OB_wap2_down_sum{suffix}']
    
    return df


def nru_features(stock):
    # Read in order book for stock
    book_df = pd.read_parquet(f"/kaggle/input/optiver-realized-volatility-prediction/book_train.parquet/stock_id={stock}")
    # Read in trades for stock
    trade_df = pd.read_parquet(f"/kaggle/input/optiver-realized-volatility-prediction/trade_train.parquet/stock_id={stock}")


    book_df = pd.merge(book_df, trade_df, on=['time_id','seconds_in_bucket'], how="left")


    book_df['is_trade_second'] = np.where(book_df['price'].notnull(), 1, 0)

    book_df['is_OB_update_no_trades_ind'] = np.where(book_df['price'].isnull(), 1, 0)

    book_df['price'] = book_df['price'].fillna(method='ffill').fillna(method='bfill')
    
    # WAP
    book_df['wap1'] = (book_df['bid_price1'] * book_df['ask_size1'] +
                                book_df['ask_price1'] * book_df['bid_size1']) / (
                                       book_df['bid_size1']+ book_df['ask_size1'])
    book_df['wap2'] = (book_df['bid_price2'] * book_df['ask_size2'] +
                                book_df['ask_price2'] * book_df['bid_size2']) / (
                                       book_df['bid_size2']+ book_df['ask_size2'])
    book_df['wap3'] = (book_df['bid_price1'] * book_df['bid_size1'] + 
                            book_df['ask_price1'] * book_df['ask_size1']) / (
                                book_df['bid_size1'] + book_df['ask_size1'])
    book_df['wap4'] = (book_df['bid_price2'] * book_df['bid_size2'] + 
                            book_df['ask_price2'] * book_df['ask_size2']) / (
                                book_df['bid_size2'] + book_df['ask_size2'])
    # Calculate wap balance
    book_df['wap_balance'] = abs(book_df['wap1'] - book_df['wap2'])

    # Price

    book_df['price_wap1_diff_abs'] = abs(book_df['price'] - book_df['wap1'])
    book_df['price_wap2_diff_abs'] = abs(book_df['price'] - book_df['wap2'])
    book_df['price_wap3_diff_abs'] = abs(book_df['price'] - book_df['wap3'])
    book_df['price_wap4_diff_abs'] = abs(book_df['price'] - book_df['wap4'])

    book_df['is_price_wap1_above'] = np.where((book_df['price'] - book_df['wap1'])>0, 1, 0)
    book_df['is_price_wap2_above'] = np.where((book_df['price'] - book_df['wap2'])>0, 1, 0)
    book_df['is_price_wap3_above'] = np.where((book_df['price'] - book_df['wap3'])>0, 1, 0)
    book_df['is_price_wap4_above'] = np.where((book_df['price'] - book_df['wap4'])>0, 1, 0)

    book_df['is_price_wap1_below'] = np.where((book_df['price'] - book_df['wap1'])<0, 1, 0)
    book_df['is_price_wap2_below'] = np.where((book_df['price'] - book_df['wap2'])<0, 1, 0)
    book_df['is_price_wap3_below'] = np.where((book_df['price'] - book_df['wap3'])<0, 1, 0)
    book_df['is_price_wap4_below'] = np.where((book_df['price'] - book_df['wap4'])<0, 1, 0)


    book_df['price_cross_wap1'] = ((book_df['is_price_wap1_above'].diff() != 0))
    book_df['price_cross_wap2'] = ((book_df['is_price_wap2_above'].diff() != 0))
    book_df['price_cross_wap3'] = ((book_df['is_price_wap3_above'].diff() != 0))
    book_df['price_cross_wap4'] = ((book_df['is_price_wap4_above'].diff() != 0))


    book_df['wap1_std10'] = book_df['wap1'].rolling(10).std(ddof=0)
    book_df['wap1_std20'] = book_df['wap1'].rolling(20).std(ddof=0)
    book_df['wap1_std50'] = book_df['wap1'].rolling(50).std(ddof=0)
    
    book_df['wap1_MA10'] = book_df['wap1'].rolling(10).mean()
    book_df['wap1_MA20'] = book_df['wap1'].rolling(20).mean()
    book_df['wap1_MA50'] = book_df['wap1'].rolling(50).mean()

    book_df['wap1_BOLU10'] = book_df['wap1_MA10'] + 2*book_df['wap1_std10']
    book_df['wap1_BOLD10'] = book_df['wap1_MA10'] - 2*book_df['wap1_std10']

    book_df['wap1_BOLU20'] = book_df['wap1_MA20'] + 2*book_df['wap1_std20']
    book_df['wap1_BOLD20'] = book_df['wap1_MA20'] - 2*book_df['wap1_std20']

    book_df['wap1_BOLU50'] = book_df['wap1_MA50'] + 2*book_df['wap1_std50']
    book_df['wap1_BOLD50'] = book_df['wap1_MA50'] - 2*book_df['wap1_std50']


    book_df['price_std20'] = book_df['price'].rolling(10).std(ddof=0)
    book_df['price_std10'] = book_df['price'].rolling(20).std(ddof=0)
    book_df['price_std50'] = book_df['price'].rolling(50).std(ddof=0)
    
    book_df['price_MA10'] = book_df['price'].rolling(10).mean()
    book_df['price_MA20'] = book_df['price'].rolling(20).mean()
    book_df['price_MA50'] = book_df['price'].rolling(50).mean()

    book_df['price_BOLU10'] = book_df['price_MA10'] + 2*book_df['price_std10']
    book_df['price_BOLD10'] = book_df['price_MA10'] - 2*book_df['price_std10']

    book_df['price_BOLU20'] = book_df['price_MA20'] + 2*book_df['price_std20']
    book_df['price_BOLD20'] = book_df['price_MA20'] - 2*book_df['price_std20']

    book_df['price_BOLU50'] = book_df['price_MA50'] + 2*book_df['price_std50']
    book_df['price_BOLD50'] = book_df['price_MA50'] - 2*book_df['price_std50']
    

    book_df['wap1_MA10_diff_abs'] = abs(book_df['wap1'] - book_df['wap1_MA10'])
    book_df['is_wap1_above_MA10'] = np.where((book_df['wap1'] - book_df['wap1_MA10'])>0, 1, 0)
    book_df['wap1_cross_MA10'] = ((book_df['is_wap1_above_MA10'].diff() != 0))

    book_df['wap1_MA20_diff_abs'] = abs(book_df['wap1'] - book_df['wap1_MA20'])
    book_df['is_wap1_above_MA20'] = np.where((book_df['wap1'] - book_df['wap1_MA20'])>0, 1, 0)
    book_df['wap1_cross_MA20'] = ((book_df['is_wap1_above_MA20'].diff() != 0))

    book_df['wap1_MA50_diff_abs'] = abs(book_df['wap1'] - book_df['wap1_MA50'])
    book_df['is_wap1_above_MA50'] = np.where((book_df['wap1'] - book_df['wap1_MA50'])>0, 1, 0)
    book_df['wap1_cross_MA50'] = ((book_df['is_wap1_above_MA50'].diff() != 0))

    book_df['wap1_BOLU10_diff_abs'] = abs(book_df['wap1'] - book_df['wap1_BOLU10'])
    book_df['is_wap1_above_BOLU10'] = np.where((book_df['wap1'] - book_df['wap1_BOLU10'])>0, 1, 0)
    book_df['wap1_cross_BOLU10'] = ((book_df['is_wap1_above_BOLU10'].diff() != 0))

    book_df['wap1_BOLU20_diff_abs'] = abs(book_df['wap1'] - book_df['wap1_BOLU20'])
    book_df['is_wap1_above_BOLU20'] = np.where((book_df['wap1'] - book_df['wap1_BOLU20'])>0, 1, 0)
    book_df['wap1_cross_BOLU20'] = ((book_df['is_wap1_above_BOLU20'].diff() != 0))

    book_df['wap1_BOLU50_diff_abs'] = abs(book_df['wap1'] - book_df['wap1_BOLU50'])
    book_df['is_wap1_above_BOLU50'] = np.where((book_df['wap1'] - book_df['wap1_BOLU50'])>0, 1, 0)
    book_df['wap1_cross_BOLU50'] = ((book_df['is_wap1_above_BOLU50'].diff() != 0))

    book_df['wap1_BOLD10_diff_abs'] = abs(book_df['wap1'] - book_df['wap1_BOLD10'])
    book_df['is_wap1_below_BOLD10'] = np.where((book_df['wap1'] - book_df['wap1_BOLD10'])<0, 1, 0)
    book_df['wap1_cross_BOLD10'] = ((book_df['is_wap1_below_BOLD10'].diff() != 0))

    book_df['wap1_BOLD20_diff_abs'] = abs(book_df['wap1'] - book_df['wap1_BOLD20'])
    book_df['is_wap1_below_BOLD20'] = np.where((book_df['wap1'] - book_df['wap1_BOLD20'])<0, 1, 0)
    book_df['wap1_cross_BOLD20'] = ((book_df['is_wap1_below_BOLD20'].diff() != 0))

    book_df['wap1_BOLD50_diff_abs'] = abs(book_df['wap1'] - book_df['wap1_BOLD50'])
    book_df['is_wap1_below_BOLD50'] = np.where((book_df['wap1'] - book_df['wap1_BOLD50'])<0, 1, 0)
    book_df['wap1_cross_BOLD50'] = ((book_df['is_wap1_below_BOLD50'].diff() != 0))

    book_df['price_MA10_diff_abs'] = abs(book_df['price'] - book_df['price_MA10'])
    book_df['is_price_above_MA10'] = np.where((book_df['price'] - book_df['price_MA10'])>0, 1, 0)
    book_df['price_cross_MA10'] = ((book_df['is_price_above_MA10'].diff() != 0))

    book_df['price_MA20_diff_abs'] = abs(book_df['price'] - book_df['price_MA20'])
    book_df['is_price_above_MA20'] = np.where((book_df['price'] - book_df['price_MA20'])>0, 1, 0)
    book_df['price_cross_MA20'] = ((book_df['is_price_above_MA20'].diff() != 0))

    book_df['price_MA50_diff_abs'] = abs(book_df['price'] - book_df['price_MA50'])
    book_df['is_price_above_MA50'] = np.where((book_df['price'] - book_df['price_MA50'])>0, 1, 0)
    book_df['price_cross_MA50'] = ((book_df['is_price_above_MA50'].diff() != 0))

    book_df['price_BOLU10_diff_abs'] = abs(book_df['price'] - book_df['price_BOLU10'])
    book_df['is_price_above_BOLU10'] = np.where((book_df['price'] - book_df['price_BOLU10'])>0, 1, 0)
    book_df['price_cross_BOLU10'] = ((book_df['is_price_above_BOLU10'].diff() != 0))

    book_df['price_BOLU20_diff_abs'] = abs(book_df['price'] - book_df['price_BOLU20'])
    book_df['is_price_above_BOLU20'] = np.where((book_df['price'] - book_df['price_BOLU20'])>0, 1, 0)
    book_df['price_cross_BOLU20'] = ((book_df['is_price_above_BOLU20'].diff() != 0))

    book_df['price_BOLU50_diff_abs'] = abs(book_df['price'] - book_df['price_BOLU50'])
    book_df['is_price_above_BOLU50'] = np.where((book_df['price'] - book_df['price_BOLU50'])>0, 1, 0)
    book_df['price_cross_BOLU50'] = ((book_df['is_price_above_BOLU50'].diff() != 0))

    book_df['price_BOLD10_diff_abs'] = abs(book_df['price'] - book_df['price_BOLD10'])
    book_df['is_price_below_BOLD10'] = np.where((book_df['price'] - book_df['price_BOLD10'])<0, 1, 0)
    book_df['price_cross_BOLD10'] = ((book_df['is_price_below_BOLD10'].diff() != 0))

    book_df['price_BOLD20_diff_abs'] = abs(book_df['price'] - book_df['price_BOLD20'])
    book_df['is_price_below_BOLD20'] = np.where((book_df['price'] - book_df['price_BOLD20'])<0, 1, 0)
    book_df['price_cross_BOLD20'] = ((book_df['is_price_below_BOLD20'].diff() != 0))

    book_df['price_BOLD50_diff_abs'] = abs(book_df['price'] - book_df['price_BOLD50'])
    book_df['is_price_below_BOLD50'] = np.where((book_df['price'] - book_df['price_BOLD50'])<0, 1, 0)
    book_df['price_cross_BOLD50'] = ((book_df['is_price_below_BOLD50'].diff() != 0))


    book_df['is_wap1_MA10_above_MA20'] = np.where((book_df['wap1_MA10'] - book_df['wap1_MA20'])>0, 1, 0)
    book_df['wap1_MA10_cross_MA20'] = ((book_df['is_wap1_MA10_above_MA20'].diff() != 0))

    book_df['is_wap1_MA20_above_MA50'] = np.where((book_df['wap1_MA20'] - book_df['wap1_MA50'])>0, 1, 0)
    book_df['wap1_MA20_cross_MA50'] = ((book_df['is_wap1_MA20_above_MA50'].diff() != 0))

    book_df['is_price_MA10_above_MA20'] = np.where((book_df['price_MA10'] - book_df['price_MA20'])>0, 1, 0)
    book_df['price_MA10_cross_MA20'] = ((book_df['is_price_MA10_above_MA20'].diff() != 0))

    book_df['is_price_MA20_above_MA50'] = np.where((book_df['price_MA20'] - book_df['price_MA50'])>0, 1, 0)
    book_df['price_MA20_cross_MA50'] = ((book_df['is_price_MA20_above_MA50'].diff() != 0))
    
    
    # Spread
    book_df['spread'] = (book_df['ask_price1'] / book_df['bid_price1']) - 1
    book_df['spread2'] = (book_df['ask_price2'] / book_df['bid_price2']) - 1
    book_df['price_spread'] = (book_df['ask_price1'] - book_df['bid_price1']) / ((book_df['ask_price1'] + book_df['bid_price1']) / 2)
    book_df['price_spread2'] = (book_df['ask_price2'] - book_df['bid_price2']) / ((book_df['ask_price2'] + book_df['bid_price2']) / 2)
    book_df['bid_spread'] = book_df['bid_price1'] - book_df['bid_price2']
    book_df['ask_spread'] = book_df['ask_price1'] - book_df['ask_price2']
    book_df["bid_ask_spread"] = abs(book_df['bid_spread'] - book_df['ask_spread'])
    book_df['total_volume'] = (book_df['ask_size1'] + book_df['ask_size2']) + (book_df['bid_size1'] + book_df['bid_size2'])
    book_df['volume_imbalance'] = abs((book_df['ask_size1'] + book_df['ask_size2']) - (book_df['bid_size1'] + book_df['bid_size2']))

    # Bid-ask diff
    book_df['bid_ask_diff'] = (book_df['ask_price1'] - book_df['bid_price1'])
    book_df['bid_ask_diff2'] = (book_df['ask_price2'] - book_df['bid_price2'])
    book_df['bid_ask_diff_wap_ratio'] = (book_df['ask_price1'] - book_df['bid_price1']) / book_df['wap1']
    book_df['bid_ask_diff_wap2_ratio'] = (book_df['ask_price1'] - book_df['bid_price1']) / book_df['wap2']
    book_df['bid_ask_diff2_wap2_ratio'] = (book_df['ask_price2'] - book_df['bid_price2']) / book_df['wap2']
    
    # Log return
    book_df['log_return1'] = book_df.groupby(['time_id'])['wap1'].apply(log_return)
    book_df['log_return2'] = book_df.groupby(['time_id'])['wap2'].apply(log_return)
    book_df['log_return3'] = book_df.groupby(['time_id'])['wap3'].apply(log_return)
    book_df['log_return4'] = book_df.groupby(['time_id'])['wap4'].apply(log_return)
    
    # Time based
    book_df['OB_static_seconds'] = book_df['seconds_in_bucket'] - book_df['seconds_in_bucket'].shift()
    book_df['OB_static_seconds'] = np.where(book_df['OB_static_seconds'].isnull(), 0 , book_df['OB_static_seconds'])

    book_df['is_OB_dynamic_second'] = np.where(book_df['OB_static_seconds'].isin([0, 1]), 1 , 0)

    book_df['OB_dynamic_group'] = ((book_df['is_OB_dynamic_second'].diff() != 0) & (book_df['is_OB_dynamic_second']!=0)).cumsum()
    book_df['OB_dynamic_group'] = np.where(book_df['is_OB_dynamic_second']==0, 0, book_df['OB_dynamic_group'])

    book_df['temp_bid_price_diff'] = (book_df['bid_price1'].diff())
    book_df['temp_bid_price_diff_abs'] = np.abs(book_df['bid_price1'].diff())

    book_df['temp_ask_price_diff'] = (book_df['ask_price1'].diff())
    book_df['temp_ask_price_diff_abs'] = np.abs(book_df['ask_price1'].diff())

    book_df['temp_wap1_diff'] = (book_df['wap1'].diff())
    book_df['temp_wap1_diff_abs'] = np.abs(book_df['wap1'].diff())

    book_df['OB_dynamic_seconds'] = book_df['is_OB_dynamic_second'].groupby(book_df['is_OB_dynamic_second'].eq(0).cumsum()).cumsum().tolist()

    book_df['OB_bid_price1_change'] = book_df['bid_price1'].diff().fillna(0)
    book_df['OB_bid_price1_change_abs'] = np.abs(book_df['bid_price1'].diff().fillna(0))
    book_df['is_OB_bid_price1_same'] = np.where(book_df['bid_price1'].diff()==0, 1, 0)
    book_df['is_OB_bid_price1_change'] = np.where(book_df['bid_price1'].diff()!=0, 1, 0)
    book_df['is_OB_bid_price1_up'] = np.where(book_df['bid_price1'].diff()>0, 1, 0)
    book_df['is_OB_bid_price1_down'] = np.where(book_df['bid_price1'].diff()<0, 1, 0)

    book_df['OB_bid_price2_change'] = book_df['bid_price2'].diff().fillna(0)
    book_df['OB_bid_price2_change_abs'] = np.abs(book_df['bid_price2'].diff().fillna(0))
    book_df['is_OB_bid_price2_same'] = np.where(book_df['bid_price2'].diff()==0, 1, 0)
    book_df['is_OB_bid_price2_change'] = np.where(book_df['bid_price2'].diff()!=0, 1, 0)
    book_df['is_OB_bid_price2_up'] = np.where(book_df['bid_price2'].diff()>0, 1, 0)
    book_df['is_OB_bid_price2_down'] = np.where(book_df['bid_price2'].diff()<0, 1, 0)

    book_df['OB_ask_price1_change'] = book_df['ask_price1'].diff().fillna(0)
    book_df['OB_ask_price1_change_abs'] = np.abs(book_df['ask_price1'].diff().fillna(0))
    book_df['is_OB_ask_price1_same'] = np.where(book_df['ask_price1'].diff()==0, 1, 0)
    book_df['is_OB_ask_price1_change'] = np.where(book_df['ask_price1'].diff()!=0, 1, 0)
    book_df['is_OB_ask_price1_up'] = np.where(book_df['ask_price1'].diff()>0, 1, 0)
    book_df['is_OB_ask_price1_down'] = np.where(book_df['ask_price1'].diff()<0, 1, 0)

    book_df['OB_ask_price2_change'] = book_df['ask_price2'].diff().fillna(0)
    book_df['OB_ask_price2_change_abs'] = np.abs(book_df['ask_price2'].diff().fillna(0))
    book_df['is_OB_ask_price2_same'] = np.where(book_df['ask_price2'].diff()==0, 1, 0)
    book_df['is_OB_ask_price2_change'] = np.where(book_df['ask_price2'].diff()!=0, 1, 0)
    book_df['is_OB_ask_price2_up'] = np.where(book_df['ask_price2'].diff()>0, 1, 0)
    book_df['is_OB_ask_price2_down'] = np.where(book_df['ask_price2'].diff()<0, 1, 0)

    book_df['OB_wap1_change'] = book_df['wap1'].diff().fillna(0)
    book_df['OB_wap1_change_abs'] = np.abs(book_df['wap1'].diff().fillna(0))
    book_df['is_OB_wap1_same'] = np.where(book_df['wap1'].diff()==0, 1, 0)
    book_df['is_OB_wap1_change'] = np.where(book_df['wap1'].diff()!=0, 1, 0)
    book_df['is_OB_wap1_up'] = np.where(book_df['wap1'].diff()>0, 1, 0)
    book_df['is_OB_wap1_down'] = np.where(book_df['wap1'].diff()<0, 1, 0)

    book_df['OB_wap2_change'] = book_df['wap2'].diff().fillna(0)
    book_df['OB_wap2_change_abs'] = np.abs(book_df['wap2'].diff().fillna(0))
    book_df['is_OB_wap2_same'] = np.where(book_df['wap2'].diff()==0, 1, 0)
    book_df['is_OB_wap2_change'] = np.where(book_df['wap2'].diff()!=0, 1, 0)
    book_df['is_OB_wap2_up'] = np.where(book_df['wap2'].diff()>0, 1, 0)
    book_df['is_OB_wap2_down'] = np.where(book_df['wap2'].diff()<0, 1, 0)

    book_df['OB_wap3_change'] = book_df['wap3'].diff().fillna(0)
    book_df['OB_wap3_change_abs'] = np.abs(book_df['wap3'].diff().fillna(0))
    book_df['is_OB_wap3_same'] = np.where(book_df['wap3'].diff()==0, 1, 0)
    book_df['is_OB_wap3_change'] = np.where(book_df['wap3'].diff()!=0, 1, 0)
    book_df['is_OB_wap3_up'] = np.where(book_df['wap3'].diff()>0, 1, 0)
    book_df['is_OB_wap3_down'] = np.where(book_df['wap3'].diff()<0, 1, 0)

    book_df['OB_wap4_change'] = book_df['wap4'].diff().fillna(0)
    book_df['OB_wap4_change_abs'] = np.abs(book_df['wap4'].diff().fillna(0))
    book_df['is_OB_wap4_same'] = np.where(book_df['wap4'].diff()==0, 1, 0)
    book_df['is_OB_wap4_change'] = np.where(book_df['wap4'].diff()!=0, 1, 0)
    book_df['is_OB_wap4_up'] = np.where(book_df['wap4'].diff()>0, 1, 0)
    book_df['is_OB_wap4_down'] = np.where(book_df['wap4'].diff()<0, 1, 0)


    book_df['temp_bid_price1_mean'] =  np.mean(book_df['bid_price1'])
    book_df['temp_ask_price1_mean'] =  np.mean(book_df['ask_price1'])
    book_df['temp_wap1_mean'] =  np.mean(book_df['wap1'])

    book_df['temp_bid_price1_std'] =  np.std(book_df['bid_price1'])
    book_df['temp_ask_price1_std'] =  np.std(book_df['ask_price1'])
    book_df['temp_wap1_std'] =  np.std(book_df['wap1'])

    book_df['is_bid_price1_above_mean'] = np.where(book_df['bid_price1'] > book_df['temp_bid_price1_mean'], 1, 0)
    book_df['is_bid_price1_below_mean'] = np.where(book_df['bid_price1'] < book_df['temp_bid_price1_mean'], 1, 0)
    book_df['bid_price1_cross_mean'] = ((book_df['is_bid_price1_above_mean'].diff() != 0))
    book_df['bid_price1_num_std'] = np.where(np.abs((book_df['bid_price1'] - book_df['temp_bid_price1_mean'])/book_df['temp_bid_price1_std']), 1, 0)

    book_df['is_ask_price1_above_mean'] = np.where(book_df['ask_price1'] > book_df['temp_ask_price1_mean'], 1, 0)
    book_df['is_ask_price1_below_mean'] = np.where(book_df['ask_price1'] < book_df['temp_ask_price1_mean'], 1, 0)
    book_df['ask_price1_cross_mean'] = ((book_df['is_ask_price1_above_mean'].diff() != 0))
    book_df['ask_price1_num_std'] = np.where(np.abs((book_df['ask_price1'] - book_df['temp_ask_price1_mean'])/book_df['temp_ask_price1_std']), 1, 0)

    book_df['is_wap1_above_mean'] = np.where(book_df['wap1'] > book_df['temp_wap1_mean'], 1, 0)
    book_df['is_wap1_below_mean'] = np.where(book_df['wap1'] < book_df['temp_wap1_mean'], 1, 0)
    book_df['wap1_cross_mean'] = ((book_df['is_wap1_above_mean'].diff() != 0))
    book_df['wap1_num_std'] = np.where(np.abs((book_df['wap1'] - book_df['temp_wap1_mean'])/book_df['temp_wap1_std']), 1, 0)

    book_df['OB_bid_size1_change'] = book_df['bid_size1'].diff().fillna(0)
    book_df['OB_bid_size1_change_abs'] = np.abs(book_df['bid_size1'].diff().fillna(0))
    book_df['is_OB_bid_size1_same'] = np.where(book_df['bid_size1'].diff()==0, 1, 0)
    book_df['is_OB_bid_size1_change'] = np.where(book_df['bid_size1'].diff()!=0, 1, 0)
    book_df['is_OB_bid_size1_up'] = np.where(book_df['bid_size1'].diff()>0, 1, 0)
    book_df['is_OB_bid_size1_down'] = np.where(book_df['bid_size1'].diff()<0, 1, 0)

    book_df['OB_bid_size2_change'] = book_df['bid_size2'].diff().fillna(0)
    book_df['OB_bid_size2_change_abs'] = np.abs(book_df['bid_size2'].diff().fillna(0))
    book_df['is_OB_bid_size2_same'] = np.where(book_df['bid_size2'].diff()==0, 1, 0)
    book_df['is_OB_bid_size2_change'] = np.where(book_df['bid_size2'].diff()!=0, 1, 0)
    book_df['is_OB_bid_size2_up'] = np.where(book_df['bid_size2'].diff()>0, 1, 0)
    book_df['is_OB_bid_size2_down'] = np.where(book_df['bid_size2'].diff()<0, 1, 0)

    book_df['OB_ask_size1_change'] = book_df['ask_size1'].diff().fillna(0)
    book_df['OB_ask_size1_change_abs'] = np.abs(book_df['ask_size1'].diff().fillna(0))
    book_df['is_OB_ask_size1_same'] = np.where(book_df['ask_size1'].diff()==0, 1, 0)
    book_df['is_OB_ask_size1_change'] = np.where(book_df['ask_size1'].diff()!=0, 1, 0)
    book_df['is_OB_ask_size1_up'] = np.where(book_df['ask_size1'].diff()>0, 1, 0)
    book_df['is_OB_ask_size1_down'] = np.where(book_df['ask_size1'].diff()<0, 1, 0)

    book_df['OB_ask_size2_change'] = book_df['ask_size2'].diff().fillna(0)
    book_df['OB_ask_size2_change_abs'] = np.abs(book_df['ask_size2'].diff().fillna(0))
    book_df['is_OB_ask_size2_same'] = np.where(book_df['ask_size2'].diff()==0, 1, 0)
    book_df['is_OB_ask_size2_change'] = np.where(book_df['ask_size2'].diff()!=0, 1, 0)
    book_df['is_OB_ask_size2_up'] = np.where(book_df['ask_size2'].diff()>0, 1, 0)
    book_df['is_OB_ask_size2_down'] = np.where(book_df['ask_size2'].diff()<0, 1, 0)


    # Size
    book_df['total_book_volume'] = book_df['ask_size1'] + book_df['ask_size2'] + book_df['bid_size1'] + book_df['bid_size2']
    book_df['total_bid_volume'] = book_df['bid_size1'] + book_df['bid_size2']
    book_df['total_ask_volume'] = book_df['ask_size1'] + book_df['ask_size2']
    
    book_df['orderSize_to_totalbook_ratio'] = np.where((book_df['size'].isnull()) | (book_df['total_book_volume']==0), 0, book_df['size'] / book_df['total_book_volume'])

    book_df['orderSize_to_totalside_ratio'] = np.where((book_df['size'].isnull()) | (book_df['total_book_volume']==0), 0, 
                                                    np.where(book_df['wap1'] > book_df['price'], book_df['size'] / book_df['total_bid_volume'], book_df['size'] / book_df['total_ask_volume']))


    book_df['temp_total_book_volume_shift'] = book_df['total_book_volume'].shift()
    book_df['temp_total_bid_volume_shift'] = book_df['total_bid_volume'].shift()
    book_df['temp_total_ask_volume_shift'] = book_df['total_ask_volume'].shift()


    book_df['orderSize_to_totalbook_shift_ratio'] = np.where((book_df['size'].isnull()) | (book_df['temp_total_book_volume_shift']==0), 0, book_df['size'] / book_df['temp_total_book_volume_shift'])

    book_df['orderSize_to_totalside_shift_ratio'] = np.where((book_df['size'].isnull()) | (book_df['temp_total_book_volume_shift']==0), 0, 
                                                    np.where(book_df['wap1'] > book_df['price'], book_df['size'] / book_df['temp_total_bid_volume_shift'], book_df['size'] / book_df['temp_total_ask_volume_shift']))
    
    
    book_df['avg_trade_size'] = np.where((book_df['size'].isnull()), 0, book_df['size'] / book_df['order_count'])

    book_df['market_cap_volume'] = np.where((book_df['size'].isnull()), 0, book_df['size'] * book_df['price'])

    
    dropCols = [col for col in book_df.columns if 'temp' in col ]
    book_df.drop(columns=dropCols, inplace=True)

    return book_df



                                                             
def ru_features(book_df, stock, since_second=None):
    
#     import gc

    if since_second:
        book_df = book_df[book_df['seconds_in_bucket']>=since_second].copy()

    book_df['temp_bid_price_diff'] = (book_df['bid_price1'].diff())
    book_df['temp_bid_price_diff_abs'] = np.abs(book_df['bid_price1'].diff())

    book_df['temp_ask_price_diff'] = (book_df['ask_price1'].diff())
    book_df['temp_ask_price_diff_abs'] = np.abs(book_df['ask_price1'].diff())

    book_df['temp_wap1_diff'] = (book_df['wap1'].diff())
    book_df['temp_wap1_diff_abs'] = np.abs(book_df['wap1'].diff())
    
    temp1 = book_df[['OB_dynamic_group','is_OB_dynamic_second','temp_bid_price_diff','temp_bid_price_diff_abs','temp_ask_price_diff','temp_ask_price_diff_abs',
              'temp_wap1_diff','temp_wap1_diff_abs']].copy()
    temp1 = temp1.groupby(['OB_dynamic_group']).agg(
                                                    OB_tot_dynamic_seconds=pd.NamedAgg(column='is_OB_dynamic_second', aggfunc=sum),
                                                    
                                                    OB_tot_bid_price_change=pd.NamedAgg(column='temp_bid_price_diff', aggfunc=sum),
                                                    OB_max_bid_price_change=pd.NamedAgg(column='temp_bid_price_diff', aggfunc=max),
                                                    OB_std_bid_price_change=pd.NamedAgg(column='temp_bid_price_diff', aggfunc="std"),
        
                                                    OB_tot_bid_price_change_abs=pd.NamedAgg(column='temp_bid_price_diff_abs', aggfunc=sum),
                                                    OB_max_bid_price_change_abs=pd.NamedAgg(column='temp_bid_price_diff_abs', aggfunc=max),
                                                    OB_std_bid_price_change_abs=pd.NamedAgg(column='temp_bid_price_diff_abs', aggfunc="std"),
        
                                                    OB_tot_ask_price_change=pd.NamedAgg(column='temp_ask_price_diff', aggfunc=sum),
                                                    OB_max_ask_price_change=pd.NamedAgg(column='temp_ask_price_diff', aggfunc=max),
                                                    OB_std_ask_price_change=pd.NamedAgg(column='temp_ask_price_diff', aggfunc="std"),
        
                                                    OB_tot_ask_price_change_abs=pd.NamedAgg(column='temp_ask_price_diff_abs', aggfunc=sum),
                                                    OB_max_ask_price_change_abs=pd.NamedAgg(column='temp_ask_price_diff_abs', aggfunc=max),
                                                    OB_std_ask_price_change_abs=pd.NamedAgg(column='temp_ask_price_diff_abs', aggfunc="std"),
        
                                                    OB_tot_wap1_change=pd.NamedAgg(column='temp_wap1_diff', aggfunc=sum),
                                                    OB_max_wap1_change=pd.NamedAgg(column='temp_wap1_diff', aggfunc=max),
                                                    OB_std_wap1_change=pd.NamedAgg(column='temp_wap1_diff', aggfunc="std"),
        
                                                    OB_tot_wap1_change_abs=pd.NamedAgg(column='temp_wap1_diff_abs', aggfunc=sum),
                                                    OB_max_wap1_change_abs=pd.NamedAgg(column='temp_wap1_diff_abs', aggfunc=max),
                                                    OB_std_wap1_change_abs=pd.NamedAgg(column='temp_wap1_diff_abs', aggfunc="std"),
        
                                                    ).reset_index(drop=False)

    book_df = pd.merge(book_df, temp1, on=['OB_dynamic_group'], how="left")

    dropCols = [col for col in book_df.columns if 'temp' in col ]
    book_df.drop(columns=dropCols, inplace=True)

    for col in ['OB_tot_bid_price_change','OB_max_bid_price_change','OB_std_bid_price_change','OB_tot_bid_price_change_abs',
                'OB_max_bid_price_change_abs','OB_std_bid_price_change_abs','OB_tot_ask_price_change','OB_max_ask_price_change',
                'OB_std_ask_price_change','OB_tot_ask_price_change_abs','OB_max_ask_price_change_abs','OB_std_ask_price_change_abs',
                'OB_tot_wap1_change','OB_max_wap1_change','OB_std_wap1_change','OB_tot_wap1_change_abs','OB_max_wap1_change_abs','OB_std_wap1_change_abs']:
        book_df[col] = np.where(book_df['OB_dynamic_group']==0, 0, book_df[col])
        book_df[col] = book_df[col].fillna(0)

    # Group by time ids
    temp = book_df.groupby(['time_id']).agg(

            dyn_OB_tot_bid_price_change_max=pd.NamedAgg(column='OB_tot_bid_price_change', aggfunc=np.max),
            dyn_OB_tot_bid_price_change_abs_max=pd.NamedAgg(column='OB_tot_bid_price_change_abs', aggfunc=np.max),

            dyn_OB_tot_ask_price_change_max=pd.NamedAgg(column='OB_tot_ask_price_change', aggfunc=np.max),
            dyn_OB_tot_ask_price_change_abs_max=pd.NamedAgg(column='OB_tot_ask_price_change_abs', aggfunc=np.max),

            dyn_OB_tot_wap1_change_max=pd.NamedAgg(column='OB_tot_wap1_change', aggfunc=np.max),
            dyn_OB_tot_wap1_change_abs_max=pd.NamedAgg(column='OB_tot_wap1_change_abs', aggfunc=np.max),
            
            # WAP
            wap1_sum=pd.NamedAgg(column='wap1', aggfunc=np.sum),
            wap1_std=pd.NamedAgg(column='wap1', aggfunc=np.std),
            wap1_max=pd.NamedAgg(column='wap1', aggfunc=np.max),
            wap1_min=pd.NamedAgg(column='wap1', aggfunc=np.min),
        
            wap2_sum=pd.NamedAgg(column='wap2', aggfunc=np.sum),
            wap2_std=pd.NamedAgg(column='wap2', aggfunc=np.std),
            wap2_max=pd.NamedAgg(column='wap2', aggfunc=np.max),
            wap2_min=pd.NamedAgg(column='wap2', aggfunc=np.min),
        
            wap3_sum=pd.NamedAgg(column='wap3', aggfunc=np.sum),
            wap3_std=pd.NamedAgg(column='wap3', aggfunc=np.std),
            wap3_max=pd.NamedAgg(column='wap3', aggfunc=np.max),
            wap3_min=pd.NamedAgg(column='wap3', aggfunc=np.min),
        
            wap4_sum=pd.NamedAgg(column='wap4', aggfunc=np.sum),
            wap4_std=pd.NamedAgg(column='wap4', aggfunc=np.std),
            wap4_max=pd.NamedAgg(column='wap4', aggfunc=np.max),
            wap4_min=pd.NamedAgg(column='wap4', aggfunc=np.min),
            
            # WAP balance
            wapbal_sum=pd.NamedAgg(column='wap_balance', aggfunc=np.sum),
            wapbal_std=pd.NamedAgg(column='wap_balance', aggfunc=np.std),
            wapbal_max=pd.NamedAgg(column='wap_balance', aggfunc=np.max),
            wapbal_min=pd.NamedAgg(column='wap_balance', aggfunc=np.min),
        
            # Realized volatility
            real_vol1=pd.NamedAgg(column='log_return1', aggfunc=realized_volatility),
            real_vol2=pd.NamedAgg(column='log_return2', aggfunc=realized_volatility),
            real_vol3=pd.NamedAgg(column='log_return3', aggfunc=realized_volatility),
            real_vol4=pd.NamedAgg(column='log_return4', aggfunc=realized_volatility),
            
            # Spread
            spread_sum=pd.NamedAgg(column='spread', aggfunc=np.sum),
            spread_max=pd.NamedAgg(column='spread', aggfunc=np.max),
            spread_min=pd.NamedAgg(column='spread', aggfunc=np.min),
            spread_std=pd.NamedAgg(column='spread', aggfunc=np.std),
            
            spread2_sum=pd.NamedAgg(column='spread2', aggfunc=np.sum),
            spread2_max=pd.NamedAgg(column='spread2', aggfunc=np.max),
            spread2_min=pd.NamedAgg(column='spread2', aggfunc=np.min),
            spread2_std=pd.NamedAgg(column='spread2', aggfunc=np.std),
        
            price_spread_sum=pd.NamedAgg(column='price_spread', aggfunc=np.sum),
            price_spread_max=pd.NamedAgg(column='price_spread', aggfunc=np.max),
            price_spread_min=pd.NamedAgg(column='price_spread', aggfunc=np.min),
            price_spread_std=pd.NamedAgg(column='price_spread', aggfunc=np.std),

            price_spread2_sum=pd.NamedAgg(column='price_spread2', aggfunc=np.sum),
            price_spread2_max=pd.NamedAgg(column='price_spread2', aggfunc=np.max),
            price_spread2_min=pd.NamedAgg(column='price_spread2', aggfunc=np.min),
            price_spread2_std=pd.NamedAgg(column='price_spread2', aggfunc=np.std),

            bid_spread_sum=pd.NamedAgg(column='bid_spread', aggfunc=np.sum),
            bid_spread_max=pd.NamedAgg(column='bid_spread', aggfunc=np.max),
            bid_spread_min=pd.NamedAgg(column='bid_spread', aggfunc=np.min),
            bid_spread_std=pd.NamedAgg(column='bid_spread', aggfunc=np.std),

            ask_spread_sum=pd.NamedAgg(column='ask_spread', aggfunc=np.sum),
            ask_spread_max=pd.NamedAgg(column='ask_spread', aggfunc=np.max),
            ask_spread_min=pd.NamedAgg(column='ask_spread', aggfunc=np.min),
            ask_spread_std=pd.NamedAgg(column='ask_spread', aggfunc=np.std),

            bid_ask_spread_sum=pd.NamedAgg(column='bid_ask_spread', aggfunc=np.sum),
            bid_ask_spread_max=pd.NamedAgg(column='bid_ask_spread', aggfunc=np.max),
            bid_ask_spread_min=pd.NamedAgg(column='bid_ask_spread', aggfunc=np.min),
            bid_ask_spread_std=pd.NamedAgg(column='bid_ask_spread', aggfunc=np.std),

            total_volume_sum=pd.NamedAgg(column='total_volume', aggfunc=np.sum),
            total_volume_max=pd.NamedAgg(column='total_volume', aggfunc=np.max),
            total_volume_min=pd.NamedAgg(column='total_volume', aggfunc=np.min),
            total_volume_std=pd.NamedAgg(column='total_volume', aggfunc=np.std),

            volume_imbalance_sum=pd.NamedAgg(column='volume_imbalance', aggfunc=np.sum),
            volume_imbalance_max=pd.NamedAgg(column='volume_imbalance', aggfunc=np.max),
            volume_imbalance_min=pd.NamedAgg(column='volume_imbalance', aggfunc=np.min),
            volume_imbalance_std=pd.NamedAgg(column='volume_imbalance', aggfunc=np.std),
        
            # Bid-ask diff
            bid_ask_diff_sum=pd.NamedAgg(column='bid_ask_diff', aggfunc=np.sum),
            bid_ask_diff_max=pd.NamedAgg(column='bid_ask_diff', aggfunc=np.max),
            bid_ask_diff_min=pd.NamedAgg(column='bid_ask_diff', aggfunc=np.min),
            bid_ask_diff2_sum=pd.NamedAgg(column='bid_ask_diff2', aggfunc=np.sum),
            bid_ask_diff2_max=pd.NamedAgg(column='bid_ask_diff2', aggfunc=np.max),
            bid_ask_diff2_min=pd.NamedAgg(column='bid_ask_diff2', aggfunc=np.min),

            bid_ask_diff_wap_ratio_mean=pd.NamedAgg(column='bid_ask_diff_wap_ratio', aggfunc="mean"),
            bid_ask_diff_wap2_ratio_mean=pd.NamedAgg(column='bid_ask_diff_wap2_ratio', aggfunc="mean"),
            bid_ask_diff2_wap2_ratio_mean=pd.NamedAgg(column='bid_ask_diff2_wap2_ratio', aggfunc="mean"),
        
            # Time based
            time_num_seconds=pd.NamedAgg(column='seconds_in_bucket', aggfunc="nunique"),
        
            time_OB_static_seconds_sum=pd.NamedAgg(column='OB_static_seconds', aggfunc=np.sum),
            time_OB_static_seconds_std=pd.NamedAgg(column='OB_static_seconds', aggfunc=np.std),
            time_OB_static_seconds_max=pd.NamedAgg(column='OB_static_seconds', aggfunc=np.max),

            time_OB_total_dynamic_seconds=pd.NamedAgg(column='is_OB_dynamic_second', aggfunc=np.sum),
            time_OB_total_dynamic_groups=pd.NamedAgg(column='OB_dynamic_group', aggfunc="nunique"),
            time_OB_dynamic_seconds_max=pd.NamedAgg(column='OB_tot_dynamic_seconds', aggfunc=np.max),


            # Price change
            OB_bid_price1_dollar_change_sum=pd.NamedAgg(column='OB_bid_price1_change', aggfunc=np.sum),
            OB_bid_price1_dollar_change_min=pd.NamedAgg(column='OB_bid_price1_change', aggfunc=np.min),
            OB_bid_price1_dollar_change_max=pd.NamedAgg(column='OB_bid_price1_change', aggfunc=np.max),

            OB_bid_price1_dollar_change_abs_sum=pd.NamedAgg(column='OB_bid_price1_change_abs', aggfunc=np.sum),
            OB_bid_price1_dollar_change_abs_std=pd.NamedAgg(column='OB_bid_price1_change_abs', aggfunc=np.std),
            OB_bid_price1_dollar_change_abs_max=pd.NamedAgg(column='OB_bid_price1_change_abs', aggfunc=np.max),

            OB_bid_price1_same_sum=pd.NamedAgg(column='is_OB_bid_price1_same', aggfunc=np.sum),
            OB_bid_price1_change_sum=pd.NamedAgg(column='is_OB_bid_price1_change', aggfunc=np.sum),
            OB_bid_price1_up_sum=pd.NamedAgg(column='is_OB_bid_price1_up', aggfunc=np.sum),
            OB_bid_price1_down_sum=pd.NamedAgg(column='is_OB_bid_price1_down', aggfunc=np.sum),

            OB_bid_price2_dollar_change_sum=pd.NamedAgg(column='OB_bid_price2_change', aggfunc=np.sum),
            OB_bid_price2_dollar_change_min=pd.NamedAgg(column='OB_bid_price2_change', aggfunc=np.min),
            OB_bid_price2_dollar_change_max=pd.NamedAgg(column='OB_bid_price2_change', aggfunc=np.max),

            OB_bid_price2_dollar_change_abs_sum=pd.NamedAgg(column='OB_bid_price2_change_abs', aggfunc=np.sum),
            OB_bid_price2_dollar_change_abs_std=pd.NamedAgg(column='OB_bid_price2_change_abs', aggfunc=np.std),
            OB_bid_price2_dollar_change_abs_max=pd.NamedAgg(column='OB_bid_price2_change_abs', aggfunc=np.max),

            OB_bid_price2_same_sum=pd.NamedAgg(column='is_OB_bid_price2_same', aggfunc=np.sum),
            OB_bid_price2_change_sum=pd.NamedAgg(column='is_OB_bid_price2_change', aggfunc=np.sum),
            OB_bid_price2_up_sum=pd.NamedAgg(column='is_OB_bid_price2_up', aggfunc=np.sum),
            OB_bid_price2_down_sum=pd.NamedAgg(column='is_OB_bid_price2_down', aggfunc=np.sum),

            OB_ask_price1_dollar_change_sum=pd.NamedAgg(column='OB_ask_price1_change', aggfunc=np.sum),
            OB_ask_price1_dollar_change_min=pd.NamedAgg(column='OB_ask_price1_change', aggfunc=np.min),
            OB_ask_price1_dollar_change_max=pd.NamedAgg(column='OB_ask_price1_change', aggfunc=np.max),

            OB_ask_price1_dollar_change_abs_sum=pd.NamedAgg(column='OB_ask_price1_change_abs', aggfunc=np.sum),
            OB_ask_price1_dollar_change_abs_std=pd.NamedAgg(column='OB_ask_price1_change_abs', aggfunc=np.std),
            OB_ask_price1_dollar_change_abs_max=pd.NamedAgg(column='OB_ask_price1_change_abs', aggfunc=np.max),

            OB_ask_price1_same_sum=pd.NamedAgg(column='is_OB_ask_price1_same', aggfunc=np.sum),
            OB_ask_price1_change_sum=pd.NamedAgg(column='is_OB_ask_price1_change', aggfunc=np.sum),
            OB_ask_price1_up_sum=pd.NamedAgg(column='is_OB_ask_price1_up', aggfunc=np.sum),
            OB_ask_price1_down_sum=pd.NamedAgg(column='is_OB_ask_price1_down', aggfunc=np.sum),

            OB_ask_price2_dollar_change_sum=pd.NamedAgg(column='OB_ask_price2_change', aggfunc=np.sum),
            OB_ask_price2_dollar_change_min=pd.NamedAgg(column='OB_ask_price2_change', aggfunc=np.min),
            OB_ask_price2_dollar_change_max=pd.NamedAgg(column='OB_ask_price2_change', aggfunc=np.max),

            OB_ask_price2_dollar_change_abs_sum=pd.NamedAgg(column='OB_ask_price2_change_abs', aggfunc=np.sum),
            OB_ask_price2_dollar_change_abs_std=pd.NamedAgg(column='OB_ask_price2_change_abs', aggfunc=np.std),
            OB_ask_price2_dollar_change_abs_max=pd.NamedAgg(column='OB_ask_price2_change_abs', aggfunc=np.max),

            OB_ask_price2_same_sum=pd.NamedAgg(column='is_OB_ask_price2_same', aggfunc=np.sum),
            OB_ask_price2_change_sum=pd.NamedAgg(column='is_OB_ask_price2_change', aggfunc=np.sum),
            OB_ask_price2_up_sum=pd.NamedAgg(column='is_OB_ask_price2_up', aggfunc=np.sum),
            OB_ask_price2_down_sum=pd.NamedAgg(column='is_OB_ask_price2_down', aggfunc=np.sum),

            OB_wap1_dollar_change_sum=pd.NamedAgg(column='OB_wap1_change', aggfunc=np.sum),
            OB_wap1_dollar_change_min=pd.NamedAgg(column='OB_wap1_change', aggfunc=np.min),
            OB_wap1_dollar_change_max=pd.NamedAgg(column='OB_wap1_change', aggfunc=np.max),

            OB_wap1_dollar_change_abs_sum=pd.NamedAgg(column='OB_wap1_change_abs', aggfunc=np.sum),
            OB_wap1_dollar_change_abs_std=pd.NamedAgg(column='OB_wap1_change_abs', aggfunc=np.std),
            OB_wap1_dollar_change_abs_max=pd.NamedAgg(column='OB_wap1_change_abs', aggfunc=np.max),

            OB_wap1_same_sum=pd.NamedAgg(column='is_OB_wap1_same', aggfunc=np.sum),
            OB_wap1_change_sum=pd.NamedAgg(column='is_OB_wap1_change', aggfunc=np.sum),
            OB_wap1_up_sum=pd.NamedAgg(column='is_OB_wap1_up', aggfunc=np.sum),
            OB_wap1_down_sum=pd.NamedAgg(column='is_OB_wap1_down', aggfunc=np.sum),

            OB_wap2_dollar_change_sum=pd.NamedAgg(column='OB_wap2_change', aggfunc=np.sum),
            OB_wap2_dollar_change_min=pd.NamedAgg(column='OB_wap2_change', aggfunc=np.min),
            OB_wap2_dollar_change_max=pd.NamedAgg(column='OB_wap2_change', aggfunc=np.max),

            OB_wap2_dollar_change_abs_sum=pd.NamedAgg(column='OB_wap2_change_abs', aggfunc=np.sum),
            OB_wap2_dollar_change_abs_std=pd.NamedAgg(column='OB_wap2_change_abs', aggfunc=np.std),
            OB_wap2_dollar_change_abs_max=pd.NamedAgg(column='OB_wap2_change_abs', aggfunc=np.max),

            OB_wap2_same_sum=pd.NamedAgg(column='is_OB_wap2_same', aggfunc=np.sum),
            OB_wap2_change_sum=pd.NamedAgg(column='is_OB_wap2_change', aggfunc=np.sum),
            OB_wap2_up_sum=pd.NamedAgg(column='is_OB_wap2_up', aggfunc=np.sum),
            OB_wap2_down_sum=pd.NamedAgg(column='is_OB_wap2_down', aggfunc=np.sum),

            OB_wap3_dollar_change_sum=pd.NamedAgg(column='OB_wap3_change', aggfunc=np.sum),
            OB_wap3_dollar_change_min=pd.NamedAgg(column='OB_wap3_change', aggfunc=np.min),
            OB_wap3_dollar_change_max=pd.NamedAgg(column='OB_wap3_change', aggfunc=np.max),

            OB_wap3_dollar_change_abs_sum=pd.NamedAgg(column='OB_wap3_change_abs', aggfunc=np.sum),
            OB_wap3_dollar_change_abs_std=pd.NamedAgg(column='OB_wap3_change_abs', aggfunc=np.std),
            OB_wap3_dollar_change_abs_max=pd.NamedAgg(column='OB_wap3_change_abs', aggfunc=np.max),

            OB_wap3_same_sum=pd.NamedAgg(column='is_OB_wap3_same', aggfunc=np.sum),
            OB_wap3_change_sum=pd.NamedAgg(column='is_OB_wap3_change', aggfunc=np.sum),
            OB_wap3_up_sum=pd.NamedAgg(column='is_OB_wap3_up', aggfunc=np.sum),
            OB_wap3_down_sum=pd.NamedAgg(column='is_OB_wap3_down', aggfunc=np.sum),

            OB_wap4_dollar_change_sum=pd.NamedAgg(column='OB_wap4_change', aggfunc=np.sum),
            OB_wap4_dollar_change_min=pd.NamedAgg(column='OB_wap4_change', aggfunc=np.min),
            OB_wap4_dollar_change_max=pd.NamedAgg(column='OB_wap4_change', aggfunc=np.max),

            OB_wap4_dollar_change_abs_sum=pd.NamedAgg(column='OB_wap4_change_abs', aggfunc=np.sum),
            OB_wap4_dollar_change_abs_std=pd.NamedAgg(column='OB_wap4_change_abs', aggfunc=np.std),
            OB_wap4_dollar_change_abs_max=pd.NamedAgg(column='OB_wap4_change_abs', aggfunc=np.max),

            OB_wap4_same_sum=pd.NamedAgg(column='is_OB_wap4_same', aggfunc=np.sum),
            OB_wap4_change_sum=pd.NamedAgg(column='is_OB_wap4_change', aggfunc=np.sum),
            OB_wap4_up_sum=pd.NamedAgg(column='is_OB_wap4_up', aggfunc=np.sum),
            OB_wap4_down_sum=pd.NamedAgg(column='is_OB_wap4_down', aggfunc=np.sum),

            trade_price_wap1_diff_sum=pd.NamedAgg(column='price_wap1_diff_abs', aggfunc=np.sum),
            trade_price_wap2_diff_sum=pd.NamedAgg(column='price_wap2_diff_abs', aggfunc=np.sum),
            trade_price_wap3_diff_sum=pd.NamedAgg(column='price_wap3_diff_abs', aggfunc=np.sum),
            trade_price_wap4_diff_sum=pd.NamedAgg(column='price_wap4_diff_abs', aggfunc=np.sum),

            trade_price_above_wap1_sum=pd.NamedAgg(column='is_price_wap1_above', aggfunc=np.sum),
            trade_price_above_wap2_sum=pd.NamedAgg(column='is_price_wap2_above', aggfunc=np.sum),
            trade_price_above_wap3_sum=pd.NamedAgg(column='is_price_wap3_above', aggfunc=np.sum),
            trade_price_above_wap4_sum=pd.NamedAgg(column='is_price_wap4_above', aggfunc=np.sum),

            trade_price_wap1_num_cross=pd.NamedAgg(column='price_cross_wap1', aggfunc=np.sum),
            trade_price_wap2_num_cross=pd.NamedAgg(column='price_cross_wap2', aggfunc=np.sum),
            trade_price_wap3_num_cross=pd.NamedAgg(column='price_cross_wap3', aggfunc=np.sum),
            trade_price_wap4_num_cross=pd.NamedAgg(column='price_cross_wap4', aggfunc=np.sum),

            OB_wap1_std10_max=pd.NamedAgg(column='wap1_std10', aggfunc=np.max),
            OB_wap1_std20_max=pd.NamedAgg(column='wap1_std20', aggfunc=np.max),
            OB_wap1_std50_max=pd.NamedAgg(column='wap1_std50', aggfunc=np.max),

            OB_wap1_MA10_max=pd.NamedAgg(column='wap1_MA10', aggfunc=np.max),
            OB_wap1_MA20_max=pd.NamedAgg(column='wap1_MA20', aggfunc=np.max),
            OB_wap1_MA50_max=pd.NamedAgg(column='wap1_MA50', aggfunc=np.max),

            OB_wap1_BOLU10_max=pd.NamedAgg(column='wap1_BOLU10', aggfunc=np.max),
            OB_wap1_BOLU20_max=pd.NamedAgg(column='wap1_BOLU20', aggfunc=np.max),
            OB_wap1_BOLU50_max=pd.NamedAgg(column='wap1_BOLU50', aggfunc=np.max),
            
            OB_wap1_BOLD10_min=pd.NamedAgg(column='wap1_BOLD10', aggfunc=np.min),
            OB_wap1_BOLD20_min=pd.NamedAgg(column='wap1_BOLD20', aggfunc=np.min),
            OB_wap1_BOLD50_min=pd.NamedAgg(column='wap1_BOLD50', aggfunc=np.min),

            trade_price_std10_max=pd.NamedAgg(column='price_std10', aggfunc=np.max),
            trade_price_std20_max=pd.NamedAgg(column='price_std20', aggfunc=np.max),
            trade_price_std50_max=pd.NamedAgg(column='price_std50', aggfunc=np.max),

            trade_price_MA10_max=pd.NamedAgg(column='price_MA10', aggfunc=np.max),
            trade_price_MA20_max=pd.NamedAgg(column='price_MA20', aggfunc=np.max),
            trade_price_MA50_max=pd.NamedAgg(column='price_MA50', aggfunc=np.max),

            trade_price_BOLU10_max=pd.NamedAgg(column='price_BOLU10', aggfunc=np.max),
            trade_price_BOLU20_max=pd.NamedAgg(column='price_BOLU20', aggfunc=np.max),
            trade_price_BOLU50_max=pd.NamedAgg(column='price_BOLU50', aggfunc=np.max),
            
            trade_price_BOLD10_min=pd.NamedAgg(column='price_BOLD10', aggfunc=np.min),
            trade_price_BOLD20_min=pd.NamedAgg(column='price_BOLD20', aggfunc=np.min),
            trade_price_BOLD50_min=pd.NamedAgg(column='price_BOLD50', aggfunc=np.min),

            OB_wap1_MA10_diff_max=pd.NamedAgg(column='wap1_MA10_diff_abs', aggfunc=np.max),
            OB_wap1_MA20_diff_max=pd.NamedAgg(column='wap1_MA20_diff_abs', aggfunc=np.max),
            OB_wap1_MA50_diff_max=pd.NamedAgg(column='wap1_MA50_diff_abs', aggfunc=np.max),
            OB_wap1_BOLU10_diff_max=pd.NamedAgg(column='wap1_BOLU10_diff_abs', aggfunc=np.max),
            OB_wap1_BOLU20_diff_max=pd.NamedAgg(column='wap1_BOLU20_diff_abs', aggfunc=np.max),
            OB_wap1_BOLU50_diff_max=pd.NamedAgg(column='wap1_BOLU50_diff_abs', aggfunc=np.max),
            OB_wap1_BOLD10_diff_max=pd.NamedAgg(column='wap1_BOLD10_diff_abs', aggfunc=np.max),
            OB_wap1_BOLD20_diff_max=pd.NamedAgg(column='wap1_BOLD20_diff_abs', aggfunc=np.max),
            OB_wap1_BOLD50_diff_max=pd.NamedAgg(column='wap1_BOLD50_diff_abs', aggfunc=np.max),

            OB_wap1_above_MA10_sum=pd.NamedAgg(column='is_wap1_above_MA10', aggfunc=np.sum),
            OB_wap1_above_MA20_sum=pd.NamedAgg(column='is_wap1_above_MA20', aggfunc=np.sum),
            OB_wap1_above_MA50_sum=pd.NamedAgg(column='is_wap1_above_MA50', aggfunc=np.sum),
            OB_wap1_above_BOLU10_sum=pd.NamedAgg(column='is_wap1_above_BOLU10', aggfunc=np.sum),
            OB_wap1_above_BOLU20_sum=pd.NamedAgg(column='is_wap1_above_BOLU20', aggfunc=np.sum),
            OB_wap1_above_BOLU50_sum=pd.NamedAgg(column='is_wap1_above_BOLU50', aggfunc=np.sum),
            OB_wap1_below_BOLD10_sum=pd.NamedAgg(column='is_wap1_below_BOLD10', aggfunc=np.sum),
            OB_wap1_below_BOLD20_sum=pd.NamedAgg(column='is_wap1_below_BOLD20', aggfunc=np.sum),
            OB_wap1_below_BOLD50_sum=pd.NamedAgg(column='is_wap1_below_BOLD50', aggfunc=np.sum),

            OB_wap1_MA10_num_cross=pd.NamedAgg(column='wap1_cross_MA10', aggfunc=np.sum),
            OB_wap1_MA20_num_cross=pd.NamedAgg(column='wap1_cross_MA20', aggfunc=np.sum),
            OB_wap1_MA50_num_cross=pd.NamedAgg(column='wap1_cross_MA50', aggfunc=np.sum),
            OB_wap1_BOLU10_num_cross=pd.NamedAgg(column='wap1_cross_BOLU10', aggfunc=np.sum),
            OB_wap1_BOLU20_num_cross=pd.NamedAgg(column='wap1_cross_BOLU20', aggfunc=np.sum),
            OB_wap1_BOLU50_num_cross=pd.NamedAgg(column='wap1_cross_BOLU50', aggfunc=np.sum),
            OB_wap1_BOLD10_num_cross=pd.NamedAgg(column='wap1_cross_BOLD10', aggfunc=np.sum),
            OB_wap1_BOLD20_num_cross=pd.NamedAgg(column='wap1_cross_BOLD20', aggfunc=np.sum),
            OB_wap1_BOLD50_num_cross=pd.NamedAgg(column='wap1_cross_BOLD50', aggfunc=np.sum),

            trade_price_MA10_diff_max=pd.NamedAgg(column='price_MA10_diff_abs', aggfunc=np.max),
            trade_price_MA20_diff_max=pd.NamedAgg(column='price_MA20_diff_abs', aggfunc=np.max),
            trade_price_MA50_diff_max=pd.NamedAgg(column='price_MA50_diff_abs', aggfunc=np.max),
            trade_price_BOLU10_diff_max=pd.NamedAgg(column='price_BOLU10_diff_abs', aggfunc=np.max),
            trade_price_BOLU20_diff_max=pd.NamedAgg(column='price_BOLU20_diff_abs', aggfunc=np.max),
            trade_price_BOLU50_diff_max=pd.NamedAgg(column='price_BOLU50_diff_abs', aggfunc=np.max),
            trade_price_BOLD10_diff_max=pd.NamedAgg(column='price_BOLD10_diff_abs', aggfunc=np.max),
            trade_price_BOLD20_diff_max=pd.NamedAgg(column='price_BOLD20_diff_abs', aggfunc=np.max),
            trade_price_BOLD50_diff_max=pd.NamedAgg(column='price_BOLD50_diff_abs', aggfunc=np.max),

            trade_price_above_MA10_sum=pd.NamedAgg(column='is_price_above_MA10', aggfunc=np.sum),
            trade_price_above_MA20_sum=pd.NamedAgg(column='is_price_above_MA20', aggfunc=np.sum),
            trade_price_above_MA50_sum=pd.NamedAgg(column='is_price_above_MA50', aggfunc=np.sum),
            trade_price_above_BOLU10_sum=pd.NamedAgg(column='is_price_above_BOLU10', aggfunc=np.sum),
            trade_price_above_BOLU20_sum=pd.NamedAgg(column='is_price_above_BOLU20', aggfunc=np.sum),
            trade_price_above_BOLU50_sum=pd.NamedAgg(column='is_price_above_BOLU50', aggfunc=np.sum),
            trade_price_below_BOLD10_sum=pd.NamedAgg(column='is_price_below_BOLD10', aggfunc=np.sum),
            trade_price_below_BOLD20_sum=pd.NamedAgg(column='is_price_below_BOLD20', aggfunc=np.sum),
            trade_price_below_BOLD50_sum=pd.NamedAgg(column='is_price_below_BOLD50', aggfunc=np.sum),

            trade_price_MA10_num_cross=pd.NamedAgg(column='price_cross_MA10', aggfunc=np.sum),
            trade_price_MA20_num_cross=pd.NamedAgg(column='price_cross_MA20', aggfunc=np.sum),
            trade_price_MA50_num_cross=pd.NamedAgg(column='price_cross_MA50', aggfunc=np.sum),
            trade_price_BOLU10_num_cross=pd.NamedAgg(column='price_cross_BOLU10', aggfunc=np.sum),
            trade_price_BOLU20_num_cross=pd.NamedAgg(column='price_cross_BOLU20', aggfunc=np.sum),
            trade_price_BOLU50_num_cross=pd.NamedAgg(column='price_cross_BOLU50', aggfunc=np.sum),
            trade_price_BOLD10_num_cross=pd.NamedAgg(column='price_cross_BOLD10', aggfunc=np.sum),
            trade_price_BOLD20_num_cross=pd.NamedAgg(column='price_cross_BOLD20', aggfunc=np.sum),
            trade_price_BOLD50_num_cross=pd.NamedAgg(column='price_cross_BOLD50', aggfunc=np.sum),

            OB_wap1_MA10_above_MA20_sum=pd.NamedAgg(column='is_wap1_MA10_above_MA20', aggfunc=np.sum),
            OB_wap1_MA10_MA20_num_cross=pd.NamedAgg(column='wap1_MA10_cross_MA20', aggfunc=np.sum),

            OB_wap1_MA20_above_MA50_sum=pd.NamedAgg(column='is_wap1_MA20_above_MA50', aggfunc=np.sum),
            OB_wap1_MA20_MA50_num_cross=pd.NamedAgg(column='wap1_MA20_cross_MA50', aggfunc=np.sum),

            trade_price_MA10_above_MA20_sum=pd.NamedAgg(column='is_price_MA10_above_MA20', aggfunc=np.sum),
            trade_price_MA10_MA20_num_cross=pd.NamedAgg(column='price_MA10_cross_MA20', aggfunc=np.sum),

            trade_price_MA20_above_MA50_sum=pd.NamedAgg(column='is_price_MA20_above_MA50', aggfunc=np.sum),
            trade_price_MA20_MA50_num_cross=pd.NamedAgg(column='price_MA20_cross_MA50', aggfunc=np.sum),
            

            OB_bid_price1_above_mean_sum=pd.NamedAgg(column='is_bid_price1_above_mean', aggfunc=np.sum),
            OB_bid_price1_below_mean_sum=pd.NamedAgg(column='is_bid_price1_below_mean', aggfunc=np.sum),
            OB_bid_price1_num_cross=pd.NamedAgg(column='bid_price1_cross_mean', aggfunc=np.sum),
            OB_bid_price1_std_max=pd.NamedAgg(column='bid_price1_num_std', aggfunc=np.max),

            OB_ask_price1_above_mean_sum=pd.NamedAgg(column='is_ask_price1_above_mean', aggfunc=np.sum),
            OB_ask_price1_below_mean_sum=pd.NamedAgg(column='is_ask_price1_below_mean', aggfunc=np.sum),
            OB_ask_price1_num_cross=pd.NamedAgg(column='ask_price1_cross_mean', aggfunc=np.sum),
            OB_ask_price1_std_max=pd.NamedAgg(column='ask_price1_num_std', aggfunc=np.max),

            OB_wap1_above_mean_sum=pd.NamedAgg(column='is_wap1_above_mean', aggfunc=np.sum),
            OB_wap1_below_mean_sum=pd.NamedAgg(column='is_wap1_below_mean', aggfunc=np.sum),
            OB_wap1_num_cross=pd.NamedAgg(column='wap1_cross_mean', aggfunc=np.sum),
            OB_wap1_std_max=pd.NamedAgg(column='wap1_num_std', aggfunc=np.max),

            OB_bid_size1_dollar_change_sum=pd.NamedAgg(column='OB_bid_size1_change', aggfunc=np.sum),
            OB_bid_size1_dollar_change_min=pd.NamedAgg(column='OB_bid_size1_change', aggfunc=np.min),
            OB_bid_size1_dollar_change_max=pd.NamedAgg(column='OB_bid_size1_change', aggfunc=np.max),

            OB_bid_size1_dollar_change_abs_sum=pd.NamedAgg(column='OB_bid_size1_change_abs', aggfunc=np.sum),
            OB_bid_size1_dollar_change_abs_std=pd.NamedAgg(column='OB_bid_size1_change_abs', aggfunc=np.std),
            OB_bid_size1_dollar_change_abs_max=pd.NamedAgg(column='OB_bid_size1_change_abs', aggfunc=np.max),

            OB_bid_size1_same_sum=pd.NamedAgg(column='is_OB_bid_size1_same', aggfunc=np.sum),
            OB_bid_size1_change_sum=pd.NamedAgg(column='is_OB_bid_size1_change', aggfunc=np.sum),
            OB_bid_size1_up_sum=pd.NamedAgg(column='is_OB_bid_size1_up', aggfunc=np.sum),
            OB_bid_size1_down_sum=pd.NamedAgg(column='is_OB_bid_size1_down', aggfunc=np.sum),

            OB_bid_size2_dollar_change_sum=pd.NamedAgg(column='OB_bid_size2_change', aggfunc=np.sum),
            OB_bid_size2_dollar_change_min=pd.NamedAgg(column='OB_bid_size2_change', aggfunc=np.min),
            OB_bid_size2_dollar_change_max=pd.NamedAgg(column='OB_bid_size2_change', aggfunc=np.max),

            OB_bid_size2_dollar_change_abs_sum=pd.NamedAgg(column='OB_bid_size2_change_abs', aggfunc=np.sum),
            OB_bid_size2_dollar_change_abs_std=pd.NamedAgg(column='OB_bid_size2_change_abs', aggfunc=np.std),
            OB_bid_size2_dollar_change_abs_max=pd.NamedAgg(column='OB_bid_size2_change_abs', aggfunc=np.max),

            OB_bid_size2_same_sum=pd.NamedAgg(column='is_OB_bid_size2_same', aggfunc=np.sum),
            OB_bid_size2_change_sum=pd.NamedAgg(column='is_OB_bid_size2_change', aggfunc=np.sum),
            OB_bid_size2_up_sum=pd.NamedAgg(column='is_OB_bid_size2_up', aggfunc=np.sum),
            OB_bid_size2_down_sum=pd.NamedAgg(column='is_OB_bid_size2_down', aggfunc=np.sum),

            OB_ask_size1_dollar_change_sum=pd.NamedAgg(column='OB_ask_size1_change', aggfunc=np.sum),
            OB_ask_size1_dollar_change_min=pd.NamedAgg(column='OB_ask_size1_change', aggfunc=np.min),
            OB_ask_size1_dollar_change_max=pd.NamedAgg(column='OB_ask_size1_change', aggfunc=np.max),

            OB_ask_size1_dollar_change_abs_sum=pd.NamedAgg(column='OB_ask_size1_change_abs', aggfunc=np.sum),
            OB_ask_size1_dollar_change_abs_std=pd.NamedAgg(column='OB_ask_size1_change_abs', aggfunc=np.std),
            OB_ask_size1_dollar_change_abs_max=pd.NamedAgg(column='OB_ask_size1_change_abs', aggfunc=np.max),

            OB_ask_size1_same_sum=pd.NamedAgg(column='is_OB_ask_size1_same', aggfunc=np.sum),
            OB_ask_size1_change_sum=pd.NamedAgg(column='is_OB_ask_size1_change', aggfunc=np.sum),
            OB_ask_size1_up_sum=pd.NamedAgg(column='is_OB_ask_size1_up', aggfunc=np.sum),
            OB_ask_size1_down_sum=pd.NamedAgg(column='is_OB_ask_size1_down', aggfunc=np.sum),

            OB_ask_size2_dollar_change_sum=pd.NamedAgg(column='OB_ask_size2_change', aggfunc=np.sum),
            OB_ask_size2_dollar_change_min=pd.NamedAgg(column='OB_ask_size2_change', aggfunc=np.min),
            OB_ask_size2_dollar_change_max=pd.NamedAgg(column='OB_ask_size2_change', aggfunc=np.max),

            OB_ask_size2_dollar_change_abs_sum=pd.NamedAgg(column='OB_ask_size2_change_abs', aggfunc=np.sum),
            OB_ask_size2_dollar_change_abs_std=pd.NamedAgg(column='OB_ask_size2_change_abs', aggfunc=np.std),
            OB_ask_size2_dollar_change_abs_max=pd.NamedAgg(column='OB_ask_size2_change_abs', aggfunc=np.max),

            OB_ask_size2_same_sum=pd.NamedAgg(column='is_OB_ask_size2_same', aggfunc=np.sum),
            OB_ask_size2_change_sum=pd.NamedAgg(column='is_OB_ask_size2_change', aggfunc=np.sum),
            OB_ask_size2_up_sum=pd.NamedAgg(column='is_OB_ask_size2_up', aggfunc=np.sum),
            OB_ask_size2_down_sum=pd.NamedAgg(column='is_OB_ask_size2_down', aggfunc=np.sum),

            OB_total_book_volume_sum=pd.NamedAgg(column='total_book_volume', aggfunc=np.sum),
            OB_total_book_volume_max=pd.NamedAgg(column='total_book_volume', aggfunc=np.max),
            OB_total_book_volume_std=pd.NamedAgg(column='total_book_volume', aggfunc=np.std),

            OB_total_bid_volume_sum=pd.NamedAgg(column='total_bid_volume', aggfunc=np.sum),
            OB_total_bid_volume_max=pd.NamedAgg(column='total_bid_volume', aggfunc=np.max),
            OB_total_bid_volume_std=pd.NamedAgg(column='total_bid_volume', aggfunc=np.std),

            OB_total_ask_volume_sum=pd.NamedAgg(column='total_ask_volume', aggfunc=np.sum),
            OB_total_ask_volume_max=pd.NamedAgg(column='total_ask_volume', aggfunc=np.max),
            OB_total_ask_volume_std=pd.NamedAgg(column='total_ask_volume', aggfunc=np.std),

            orderSize_to_totalbook_sum=pd.NamedAgg(column='orderSize_to_totalbook_ratio', aggfunc=np.sum),
            orderSize_to_totalbook_max=pd.NamedAgg(column='orderSize_to_totalbook_ratio', aggfunc=np.max),
            orderSize_to_totalbook_std=pd.NamedAgg(column='orderSize_to_totalbook_ratio', aggfunc=np.std),

            orderSize_to_totalbook_shift_sum=pd.NamedAgg(column='orderSize_to_totalbook_shift_ratio', aggfunc=np.sum),
            orderSize_to_totalbook_shift_max=pd.NamedAgg(column='orderSize_to_totalbook_shift_ratio', aggfunc=np.max),
            orderSize_to_totalbook_shift_std=pd.NamedAgg(column='orderSize_to_totalbook_shift_ratio', aggfunc=np.std),

            orderSize_to_totalside_sum=pd.NamedAgg(column='orderSize_to_totalside_ratio', aggfunc=np.sum),
            orderSize_to_totalside_max=pd.NamedAgg(column='orderSize_to_totalside_ratio', aggfunc=np.max),
            orderSize_to_totalside_std=pd.NamedAgg(column='orderSize_to_totalside_ratio', aggfunc=np.std),

            orderSize_to_totalside_shift_sum=pd.NamedAgg(column='orderSize_to_totalside_shift_ratio', aggfunc=np.sum),
            orderSize_to_totalside_shift_max=pd.NamedAgg(column='orderSize_to_totalside_shift_ratio', aggfunc=np.max),
            orderSize_to_totalside_shift_std=pd.NamedAgg(column='orderSize_to_totalside_shift_ratio', aggfunc=np.std),

            avg_trade_size_sum=pd.NamedAgg(column='avg_trade_size', aggfunc=np.sum),
            avg_trade_size_max=pd.NamedAgg(column='avg_trade_size', aggfunc=np.max),
            avg_trade_size_std=pd.NamedAgg(column='avg_trade_size', aggfunc=np.std),

            market_cap_volume_sum=pd.NamedAgg(column='market_cap_volume', aggfunc=np.sum),
            market_cap_volume_max=pd.NamedAgg(column='market_cap_volume', aggfunc=np.max),
            market_cap_volume_std=pd.NamedAgg(column='market_cap_volume', aggfunc=np.std),    
        
            ).reset_index(drop=False)
    
    
    book_df = temp #pd.merge(temp, realvol_df, on="time_id", how="left") 

    del temp, temp1
#     _ = gc.collect()
    
    book_df['stock_id'] = stock
    book_df = book_df.reset_index(drop=True)
    
    return book_df




