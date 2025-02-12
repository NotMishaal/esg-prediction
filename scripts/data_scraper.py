import pandas as pd
import yfinance as yf

def get_all_financial_data(tickers):
    """
    This function takes a list of ticker strings and returns a dataframe
    that contains the financial data for all of the companies in the list. 
    The data is queried from Yahoo Finance and then merged into a
    single dataframe for each company.

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
