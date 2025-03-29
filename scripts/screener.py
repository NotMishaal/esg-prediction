import pandas as pd
import yfinance as yf
from yfinance import EquityQuery


def validate_region_codes(codes):
    """
    Filters a passed list, keeping only valid country codes

    Parameters:
        codes (list): list of country codes to validate
    """
    # list of valid country codes in yahoo finance
    country_codes = [
        'ar', 'at', 'au', 'be', 'br', 'ca', 'ch', 'cl', 'cn', 'co', 'cz', 'de', 
        'dk', 'ee', 'eg', 'es', 'fi', 'fr', 'gb', 'gr', 'hk', 'hu', 'id', 'ie', 
        'il', 'in', 'is', 'it', 'jp', 'kr', 'kw', 'lk', 'lt', 'lv', 'mx', 'my', 
        'nl', 'no', 'nz', 'pe', 'ph', 'pk', 'pl', 'pt', 'qa', 'ro', 'ru', 'sa', 
        'se', 'sg', 'sr', 'th', 'tr', 'tw', 'us', 've', 'vn', 'za'
    ]

    region_codes = list(set(codes).intersection(set(country_codes)))
    region_codes.insert(0, 'region')

    return region_codes


def fetch_screened_stocks(region_codes, batch_size=250):
    """
    Fetches a list of all the stocks in the specified regions

    Parameters:
        region_codes (list): list of region to filter by
        batch_size (int): number of stocks to fetch per batch (max is 250)
    """

    offset = 0
    all_results = []
    prev_start = -1

    region_query = EquityQuery('is-in', region_codes)

    while True:
        # fetch a batch of results
        screened_stocks = yf.screen(region_query, size=batch_size, offset=offset, sortField="intradaymarketcap")
        
        # break the loop if no more results are returned
        if screened_stocks['count'] == 0:
            print("No more results to fetch.")
            break

        if screened_stocks['start'] == prev_start:
            # update the query if no new results are returned
            prev_cap = screened_stocks['quotes'][-1]['marketCap'] # get the smallest market cap from the last results

            region_query = EquityQuery('and', [
                EquityQuery('is-in', region_codes),
                EquityQuery('lte', ['intradaymarketcap', prev_cap])
            ])
            
            offset = 0
            print(f"Repetition detected at {prev_start}\nResetting with new query at market cap {prev_cap}")
            continue

        prev_start = screened_stocks['start']
        
        # append the current batch to the all_results list
        all_results.append(screened_stocks)
        
        print(f"Batch {offset // batch_size + 1} complete.")
        # increment the offset by the batch size for the next iteration
        offset += batch_size

    return all_results

def merge_screened_stocks(screening_results):
    """
    Merge the list of screening results into a single DataFrame
    
    Parameters:
        screening_results (list): List of screening results from fetch_screened_stocks
    
    Returns:
        pd.DataFrame: Merged DataFrame containing the stock data
    """
    temp_data = []
    for result in screening_results:
        temp_data.extend(result['quotes'])

    return pd.DataFrame(temp_data)

def assign_cap_category(cumulative_percentage):
    """
    Assigns a company size category based on the cumulative percentage of market capitalization

    cumulative_percentage (float): cumulative percentage of market capitalization

    Returns:
        str: one of 'Large-Cap', 'Mid-Cap', or 'Small-Cap'
    """
    if cumulative_percentage <= 70:
        return 'Large-Cap'
    elif cumulative_percentage <= 90:
        return 'Mid-Cap'
    else:
        return 'Small-Cap'
    
    
def filter_screened_stocks(screened_stocks, n_keep=50):
    """
    Filter the screened stocks to keep the top n_keep stocks from each relative company size

    Parameters:
        screened_stocks (pd.DataFrame): List of screened stock results to filter
        n_keep (int): Number of stocks to keep from each company size

    Returns:
        pd.DataFrame: The filtered DataFrame with the top n_keep stocks from each company size
    """

    screened_stocks.drop_duplicates(subset=['longName'], keep='first', inplace=True)
    filtered = screened_stocks.sort_values('marketCap', ascending=False)
    filtered.dropna(inplace=True, subset=['marketCap'])

    # calculate cumulative market cap
    filtered['cumulativeMarketCap'] = filtered['marketCap'].cumsum()
    filtered['cumulativeMarketCapPercentage'] = filtered['cumulativeMarketCap'] / filtered['marketCap'].sum() * 100

    # assign company size
    filtered['companySize'] = filtered['cumulativeMarketCapPercentage'].apply(assign_cap_category)

    # filtered = filtered.groupby('companySize').head(n_keep)  # filters the top n_keep stocks from each company size
    return filtered

def get_region_stocks(region_codes, n_keep=50, batch_size=250):
    """
    Fetches a list of stocks from the specified regions, filters them to keep the top n_keep stocks from each relative company size, and returns the filtered DataFrame

    Parameters:
        region_codes (list): List of Yahoo Finance region codes to filter by
        n_keep (int): Number of stocks to keep from each company size
        batch_size (int): Number of stocks to fetch per batch (max is 250)

    Returns:
        pd.DataFrame: The filtered DataFrame with the top n_keep stocks from each company size
    """
    region_codes = validate_region_codes(region_codes)

    screened_stocks = fetch_screened_stocks(region_codes, batch_size)
    screened_stocks = merge_screened_stocks(screened_stocks)

    return filter_screened_stocks(screened_stocks, n_keep)