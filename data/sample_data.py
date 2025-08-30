import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# --- Generate sales time series for a single product ---
def generate_product_series(start_date, days, base, trend, weekly_seasonality, noise_scale, promo_freq=60):
    """
    Generate synthetic daily sales data for a single product.

    Args:
        start_date (str): Starting date in 'YYYY-MM-DD' format.
        days (int): Number of days to simulate.
        base (float): Base sales level.
        trend (float): Linear trend across the period (positive or negative).
        weekly_seasonality (float): Weekly sales amplitude.
        noise_scale (float): Std deviation of Gaussian noise added to sales.
        promo_freq (int): Frequency (days) of promotions.

    Returns:
        pd.DataFrame: DataFrame with columns [date, sales, promo].
    """
    rng = pd.date_range(start_date, periods=days, freq='D')
    series = []

    for i, d in enumerate(rng):
        t = i
        # Linear trend (scaled over entire duration)
        trend_comp = base + trend * t / days

        # Weekly seasonality (cyclical pattern by weekday)
        weekly = weekly_seasonality * (1 + 0.2 * np.sin(2 * np.pi * (d.weekday()) / 7))

        # Annual seasonality (yearly sinusoidal variation)
        seasonal = 5 * np.sin(2 * np.pi * i / 365.25)

        # Promotional effect: spike in sales every `promo_freq` days
        promo = 0
        if (i % promo_freq) == 0:
            promo = np.random.randint(10, 40)

        # Random noise for realism
        noise = np.random.normal(0, noise_scale)

        # Final sales value (clipped at 0 to avoid negatives)
        val = max(0, trend_comp + weekly + seasonal + promo + noise)

        # Append tuple of values
        series.append((d.strftime('%Y-%m-%d'), val, promo))

    # Build DataFrame
    df = pd.DataFrame(series, columns=['date', 'sales', 'promo'])
    return df


# --- Generate sales data for multiple products ---
def generate_multi_product(days=365, n_products=5, start_date='2023-01-01'):
    """
    Generate synthetic sales datasets for multiple products.

    Each product has unique parameters for base, trend, weekly seasonality,
    noise, promo frequency, price, and marketing spend.

    Args:
        days (int): Number of days to simulate.
        n_products (int): Number of distinct products.
        start_date (str): Starting date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: Combined dataset with product-level features.
    """
    all_dfs = []

    for pid in range(1, n_products+1):
        # Randomize product-specific parameters
        base = np.random.uniform(20, 200)
        trend = np.random.uniform(-10, 50)
        weekly = np.random.uniform(5, 30)
        noise = np.random.uniform(5, 30)
        promo_freq = np.random.choice([30, 60, 90])

        # Generate series for this product
        df = generate_product_series(start_date, days, base, trend, weekly, noise, promo_freq=promo_freq)

        # Add product metadata
        df['product_id'] = f'P{pid:03d}'  # e.g. P001, P002
        df['price'] = np.round(np.random.uniform(5, 50) * (1 + 0.01 * np.random.randn(len(df))), 2)
        df['marketing_spend'] = np.round(np.random.uniform(0, 1000) * np.random.rand(len(df)), 2)

        all_dfs.append(df)

    # Combine all product data into one DataFrame
    return pd.concat(all_dfs, ignore_index=True)


# --- CLI entry point ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic product sales data")
    parser.add_argument('--output', default='data/sample_data.csv', help="Output CSV path")
    parser.add_argument('--days', type=int, default=730, help="Number of days to simulate")
    parser.add_argument('--n_products', type=int, default=10, help="Number of products to simulate")
    parser.add_argument('--start_date', default='2023-01-01', help="Simulation start date (YYYY-MM-DD)")
    args = parser.parse_args()

    # Generate dataset
    df = generate_multi_product(days=args.days, n_products=args.n_products, start_date=args.start_date)

    # Shuffle rows to mimic unstructured real-world logs
    df = df.sample(frac=1).reset_index(drop=True)

    # Write CSV
    df.to_csv(args.output, index=False)
    print(f'Wrote sample data to {args.output} — rows: {len(df)}')
