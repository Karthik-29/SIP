import pandas as pd
import matplotlib.pyplot as plt

def analyze_sip_days(csv_path):
    # Load and prepare data
    df = pd.read_csv(
        csv_path,
        parse_dates=['Date'],
        dayfirst=True,  # Critical for DD-MM-YYYY format
        thousands=',',  # Handle values like "21,731.40"
        usecols=['Date', 'Open', 'Close'],  # Only keep needed columns
        dtype={'Open': float, 'Price': float}
    )
    df = df.sort_values('Date').set_index('Date')
    
    # Validate required columns
    if not {'Open', 'Close'}.issubset(df.columns):
        raise ValueError("Data must contain both 'Open' and 'Close' columns")
    
    results = []
    
    for day in range(1, 32):
        sip_dates = []
        
        # Generate months between first data date and end date
        months = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')
        
        for month_start in months:
            # Calculate target date for current month
            month_end = month_start + pd.offsets.MonthEnd(1)
            target_date = month_start + pd.DateOffset(days=day-1)
            
            # Don't look beyond our data end date
            target_date = min(target_date, df.index.max())
            
            # Adjust if target exceeds month end
            target_date = min(target_date, month_end)
            
            # Find first valid trading date on or after target date
            valid_dates = df.index[(df.index >= target_date) & (df.index <= month_end)]
            if not valid_dates.empty:
                sip_dates.append(valid_dates[0])
        
        if len(sip_dates) < 12:  # Require at least 1 year of data
            results.append({'Day': day, 'Return': None})
            continue
        
        try:
            # Calculate returns using opening prices for purchases
            opening_prices = df.loc[sip_dates, 'Open']
            units = (1 / opening_prices).sum()
            total_investment = len(opening_prices)
            
            # Use closing price from last SIP date for valuation
            last_sip_date = sip_dates[-1]
            final_closing = df.loc[last_sip_date, 'Close']
            
            final_value = units * final_closing
            returnVal = (final_value - total_investment)
            results.append({'Day': day, 'Return': returnVal})
        except Exception as e:
            print(f"Error processing day {day}: {e}")
            results.append({'Day': day, 'Return': None})
    
    # Create results dataframe
    results_df = pd.DataFrame(results).dropna().sort_values('Return', ascending=False)
    return results_df

# Example usage
if __name__ == "__main__":
    try:
        results_df = analyze_sip_days('data.csv')
        print("\nBest SIP Days Analysis (2004-2023):")
        print(results_df.to_string(index=False))
        
        # Calculate and display statistics
        

        df = results_df.sort_values('Day')
        df = df[df['Day'] <= 29]
        mean_return = df['Return'].mean()
        std_return = df['Return'].std()
        df.to_csv('result.csv', index=False)
        print(f"\nMean Return: {mean_return:.2f}")
        print(f"Standard Deviation: {std_return:.2f}")
        print(df.to_string(index=False))
        # Plot Bar Chart
        plt.figure(figsize=(10, 5))
        plt.bar(df['Day'], df['Return'], color='skyblue')

        # Labels and Title
        plt.xlabel('Day')
        plt.ylabel('Return')
        plt.title('Return vs Day')
        plt.xticks(df['Day'])  # Ensure all days appear on x-axis
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Show Plot
        plt.show()
    except Exception as e:
        print(f"Error: {e}")