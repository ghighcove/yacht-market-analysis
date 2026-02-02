#!/usr/bin/env python3
"""
Quick Yacht Dataset Generator - Create enhanced dataset for ML training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_yacht_dataset():
    """Generate enhanced yacht market dataset"""
    print("Generating enhanced yacht market dataset...")
    
    np.random.seed(42)
    n_yachts = 1000
    
    # Base characteristics
    data = {
        'yacht_id': range(1, n_yachts + 1),
        'length_meters': np.random.normal(30, 8, n_yachts).clip(10, 60),
        'beam_meters': np.random.normal(6, 1.5, n_yachts).clip(3, 12),
        'draft_meters': np.random.normal(2.5, 0.8, n_yachts).clip(1.0, 5.0),
        'year_built': np.random.randint(1990, 2024, n_yachts),
        'engine_hours': np.random.exponential(2000, n_yachts).clip(0, 15000),
        'fuel_capacity_l': np.random.normal(1500, 500, n_yachts).clip(200, 5000),
        'water_capacity_l': np.random.normal(800, 300, n_yachts).clip(100, 3000),
        'num_cabins': np.random.randint(2, 8, n_yachts),
        'num_berths': np.random.randint(4, 16, n_yachts),
        'num_heads': np.random.randint(1, 6, n_yachts),
        'engine_power_hp': np.random.normal(800, 300, n_yachts).clip(100, 3000),
        'cruise_speed_knots': np.random.normal(20, 5, n_yachts).clip(8, 35),
        'max_speed_knots': np.random.normal(25, 6, n_yachts).clip(12, 45),
        'range_nm': np.random.normal(1500, 500, n_yachts).clip(300, 4000),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate age
    current_year = 2024
    df['age_years'] = current_year - df['year_built']
    
    # Geographic location
    locations = ['French Riviera', 'Italian Coast', 'Spanish Coast', 'Greek Islands', 'Croatia', 'Monaco', 'Sardinia']
    df['location'] = np.random.choice(locations, n_yachts)
    
    # Brand/Categories
    brands = ['Azimut', 'Ferretti', 'Sunseeker', 'Princess', 'Fairline', 'Viking', 'Hatteras', 'Bertram']
    df['brand'] = np.random.choice(brands, n_yachts)
    
    # Condition categories
    conditions = ['Excellent', 'Very Good', 'Good', 'Fair']
    df['condition'] = np.random.choice(conditions, n_yachts, p=[0.3, 0.4, 0.2, 0.1])
    
    # Calculate price based on features
    base_price = 500000  # Base price in euros
    
    # Length factor (strongest predictor)
    length_factor = (df['length_meters'] - 10) * 80000
    
    # Age factor (depreciation)
    age_factor = df['age_years'] * -15000
    
    # Brand factor (premium vs standard)
    brand_premium = {'Azimut': 200000, 'Ferretti': 250000, 'Sunseeker': 180000, 'Princess': 150000, 
                    'Fairline': 120000, 'Viking': 300000, 'Hatteras': 280000, 'Bertram': 220000}
    brand_factor = df['brand'].map(brand_premium)
    
    # Condition factor
    condition_factor = df['condition'].map({'Excellent': 200000, 'Very Good': 100000, 'Good': 0, 'Fair': -100000})
    
    # Engine/Power factor
    power_factor = df['engine_power_hp'] * 100
    
    # Location premium
    location_premium = {'French Riviera': 150000, 'Monaco': 300000, 'Italian Coast': 100000, 'Sardinia': 120000,
                       'Spanish Coast': 80000, 'Greek Islands': 60000, 'Croatia': 50000}
    location_factor = df['location'].map(location_premium)
    
    # Calculate sale price with some noise
    df['sale_price'] = (base_price + length_factor + age_factor + brand_factor + 
                        condition_factor + power_factor + location_factor + 
                        np.random.normal(0, 50000, n_yachts))
    
    # Ensure positive prices
    df['sale_price'] = df['sale_price'].clip(100000, 10000000)
    
    # Add asking price (usually higher than sale price)
    df['asking_price_eur'] = df['sale_price'] * np.random.normal(1.15, 0.1, n_yachts).clip(1.0, 1.5)
    
    # Additional derived features
    df['length_to_beam_ratio'] = df['length_meters'] / df['beam_meters']
    df['engine_hours_per_year'] = df['engine_hours'] / df['age_years'].clip(1, None)
    df['price_per_meter'] = df['sale_price'] / df['length_meters']
    df['age_to_length_ratio'] = df['age_years'] / df['length_meters']
    df['fuel_per_meter'] = df['fuel_capacity_l'] / df['length_meters']
    df['power_to_weight_ratio'] = df['engine_power_hp'] / df['length_meters']
    
    # Add some categorical derived features
    df['size_category'] = pd.cut(df['length_meters'], 
                                 bins=[0, 20, 30, 40, 100], 
                                 labels=['Small', 'Medium', 'Large', 'Superyacht'])
    df['age_category'] = pd.cut(df['age_years'], 
                                bins=[0, 5, 10, 20, 100], 
                                labels=['New', 'Modern', 'Mature', 'Classic'])
    df['speed_category'] = pd.cut(df['max_speed_knots'], 
                                  bins=[0, 20, 25, 30, 100], 
                                  labels=['Slow', 'Moderate', 'Fast', 'High-Performance'])
    
    # Add broker information
    brokers = ['Med Yacht Brokers', 'Riviera Yacht Sales', 'Mediterranean Marine', 'Luxury Yacht Group']
    df['broker'] = np.random.choice(brokers, n_yachts)
    
    # Add listing dates
    start_date = datetime(2023, 1, 1)
    df['listing_date'] = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_yachts)]
    df['days_on_market'] = np.random.randint(30, 365, n_yachts)
    df['last_seen_date'] = df['listing_date'] + pd.to_timedelta(df['days_on_market'], unit='D')
    
    # Add listing numbers
    df['listing_number'] = [f"YACHT-{i:04d}" for i in range(n_yachts)]
    
    # Currency conversions
    exchange_rates = {'USD': 1.08, 'GBP': 0.86, 'EUR': 1.0}
    df['asking_price_usd'] = df['asking_price_eur'] * exchange_rates['USD']
    df['asking_price_gbp'] = df['asking_price_eur'] * exchange_rates['GBP']
    df['sale_price_usd'] = df['sale_price'] * exchange_rates['USD']
    df['sale_price_gbp'] = df['sale_price'] * exchange_rates['GBP']
    
    # Calculate ratios
    df['sale_to_ask_ratio'] = df['sale_price'] / df['asking_price_eur']
    
    print(f"Generated dataset: {len(df)} yachts, {len(df.columns)} features")
    print(f"Price range: €{df['sale_price'].min():,.0f} - €{df['sale_price'].max():,.0f}")
    print(f"Average price: €{df['sale_price'].mean():,.0f}")
    
    return df

def main():
    """Generate and save dataset"""
    df = generate_yacht_dataset()
    
    # Save to current working directory (yacht_market_analysis)
    df.to_csv('enhanced_yacht_market_data.csv', index=False)
    
    print("Dataset saved as 'enhanced_yacht_market_data.csv'")
    
    # Basic stats
    print(f"\nDataset Summary:")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Numeric features: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"Categorical features: {len(df.select_dtypes(include=['object']).columns)}")
    
    return df

if __name__ == "__main__":
    df = main()