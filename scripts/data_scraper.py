import pandas as pd
import yfinance as yf
import concurrent.futures

def get_all_financial_data(tickers):
    """
    This function takes a list of ticker strings and returns a dataframe
    that contains the financial, cash flow and balance sheet data for all 
    of the companies in the list. 
    
    Parameters
    ----------
    tickers : list
        A list of ticker strings

    Returns
    -------
    df : pandas dataframe
        A dataframe containing the financial data for all of the companies in
        the list
    """

    company_data = []  # list to store each company's data as their own df
    
    for ticker_str in tickers:
        ticker = yf.Ticker(ticker_str)

        income_stmt = ticker.income_stmt.T.reset_index()
        balance_sheet = ticker.balance_sheet.T.reset_index()
        cash_flow = ticker.cashflow.T.reset_index()

        df_temp = income_stmt.merge(balance_sheet, on='index')
        df_temp = df_temp.merge(cash_flow, on='index')

        df_temp['Ticker'] = ticker_str
        df_temp.rename(columns={'index': 'Date'})

        company_data.append(df_temp)

    df_final = pd.concat(company_data)
    return df_final

# def scrape_esg(tickers):
#     """
#     This function takes a list of ticker strings and returns a dataframe
#     that contains the Environmental, Social, and Governance (ESG) data for
#     all of the companies in the list. The data is queried from Yahoo Finance
#     and then merged into a single dataframe.

#     Parameters
#     ----------
#     tickers : list
#         A list of ticker strings

#     Returns
#     -------
#     df : pandas dataframe
#         A dataframe containing the ESG data for all of the companies in
#         the list
#     """
    
#     company_data = []  # list to store each company's data as their own df
    
#     for ticker_str in tickers:
#         ticker = yf.Ticker(ticker_str)

#         df_temp = ticker.sustainability.T
#         df_temp['Ticker'] = ticker_str

#         company_data.append(df_temp)

#     df = pd.concat(company_data)
#     return df


def get_sustainability(ticker_str):
    """Get sustainability data for a single ticker"""
    try:
        ticker = yf.Ticker(ticker_str)
        if hasattr(ticker, 'sustainability') and ticker.sustainability is not None:
            df_temp = ticker.sustainability.T
            df_temp['Ticker'] = ticker_str
            return df_temp
    except Exception as e:
        print(f"Error fetching data for {ticker_str}: {e}")
    return None

def scrape_esg(tickers):
    # Use ThreadPoolExecutor for concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all ticker requests in parallel
        future_to_ticker = {executor.submit(get_sustainability, ticker): ticker for ticker in tickers}
        
        # Collect results as they complete
        company_data = []
        for future in concurrent.futures.as_completed(future_to_ticker):
            df = future.result()
            if df is not None:
                company_data.append(df)
    
    # Concatenate if we have data
    if company_data:
        df_final = pd.concat(company_data, ignore_index=True)
        return df_final
    else:
        return pd.DataFrame()