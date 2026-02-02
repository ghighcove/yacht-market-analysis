#!/usr/bin/env python3
"""
Yacht Data Validation System
Spot-checks for real vs synthetic data detection
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """Load and analyze the yacht dataset"""
    print("Loading yacht dataset for validation...")
    df = pd.read_csv('yacht_dataset_1000_enhanced.csv')
    print(f"Dataset loaded: {len(df)} records, {len(df.columns)} columns")
    return df

def validate_yacht_ids(df):
    """Check yacht ID patterns and uniqueness"""
    print("\nValidating Yacht IDs...")
    issues = []
    
    # Check ID patterns
    id_patterns = df['yacht_id'].value_counts()
    print(f"Unique yacht IDs: {len(id_patterns)}")
    print(f"ID pattern examples: {list(df['yacht_id'].head(10))}")
    
    # Check for sequential patterns (synthetic data indicator)
    ids = df['yacht_id'].tolist()
    numeric_ids = []
    for yacht_id in ids:
        if 'YHT' in str(yacht_id):
            try:
                numeric_part = yacht_id.replace('YHT', '')
                if numeric_part.isdigit():
                    numeric_ids.append(int(numeric_part))
            except:
                pass
    
    if numeric_ids:
        numeric_ids.sort()
        sequential_gaps = []
        for i in range(1, len(numeric_ids)):
            if numeric_ids[i] - numeric_ids[i-1] > 100:
                sequential_gaps.append((numeric_ids[i-1], numeric_ids[i]))
        
        if sequential_gaps:
            issues.append(f"Suspicious sequential gaps in yacht IDs: {sequential_gaps[:5]}")
    
    return issues

def validate_builders(df):
    """Check builder realism"""
    print("\nValidating Yacht Builders...")
    issues = []
    
    builders = df['builder'].value_counts()
    print(f"Unique builders: {len(builders)}")
    print("Top builders:")
    for builder, count in builders.head(10).items():
        print(f"  {builder}: {count} yachts")
    
    # Real yacht builders verification
    known_builders = {
        'Feadship', 'Lürssen', 'Oceanco', 'Azimut', 'Sunseeker', 
        'Ferretti', 'Princess', 'Fairline', 'Viking', 'Bertram',
        'Heesen', 'Amels', 'Lazzara', 'Rossi Navi', 'Benetti'
    }
    
    dataset_builders = set(builders.index)
    unknown_builders = dataset_builders - known_builders
    
        if unknown_builders:
            issues.append(f"Unknown/suspicious builders: {list(unknown_builders)}")
            print(f"Unknown builders found: {list(unknown_builders)}")
    
    # Check builder distribution (synthetic data often has unrealistic distribution)
    if len(builders) < 10:
        issues.append("Very few unique builders - potentially synthetic")
    
    return issues

def validate_specifications(df):
    """Check yacht specifications for realism"""
    print("\nValidating Yacht Specifications...")
    issues = []
    
    # Length validation
    length_stats = df['length_m'].describe()
    print(f"Length stats: {length_stats['min']:.1f}m - {length_stats['max']:.1f}m")
    
    # Real yacht length distribution check
    if df['length_m'].max() > 180:  # Largest superyachts rarely exceed 180m
        issues.append("Unrealistic maximum yacht length")
    
    if df['length_m'].min() < 10:  # Smallest recreational yachts ~10m
        issues.append("Unrealistic minimum yacht length")
    
    # Beam validation (should be reasonable ratio to length)
    beam_to_length = df['beam_m'] / df['length_m']
    print(f"Beam/length ratio: {beam_to_length.mean():.3f} (typical: 0.15-0.25)")
    
    if beam_to_length.mean() > 0.3 or beam_to_length.mean() < 0.1:
        issues.append("Suspicious beam-to-length ratios")
    
    # GT (Gross Tonnage) validation
    if 'gt' in df.columns:
        gt_stats = df['gt'].describe()
        print(f"GT stats: {gt_stats['min']:.0f} - {gt_stats['max']:.0f}")
        
        # GT should correlate reasonably with length
        gt_per_meter = df['gt'] / df['length_m']
        print(f"GT per meter: {gt_per_meter.mean():.1f}")
        
        if gt_per_meter.mean() > 10:  # Unreasonable GT density
            issues.append("Unrealistic GT-to-length ratios")
    
    # Year validation
    if 'year_built' in df.columns:
        year_stats = df['year_built'].describe()
        print(f"Year built: {year_stats['min']:.0f} - {year_stats['max']:.0f}")
        
        current_year = datetime.now().year
        if df['year_built'].max() > current_year + 1:
            issues.append("Future build years detected")
        
        if df['year_built'].min() < 1950:  # Very old yachts
            issues.append("Suspiciously old build years")
    
    return issues

def validate_prices(df):
    """Check price distributions for realism"""
    print("\nValidating Price Data...")
    issues = []
    
    price_col = 'price_eur' if 'price_eur' in df.columns else 'sale_price'
    price_stats = df[price_col].describe()
    print(f"Price stats (€):")
    print(f"  Min: {price_stats['min']:,.0f}")
    print(f"  Max: {price_stats['max']:,.0f}")
    print(f"  Mean: {price_stats['mean']:,.0f}")
    
    # Price per meter validation
    if 'length_m' in df.columns:
        price_per_meter = df[price_col] / df['length_m']
        ppm_stats = price_per_meter.describe()
        print(f"Price per meter: €{ppm_stats['min']:,.0f} - €{ppm_stats['max']:,.0f}")
        print(f"Average: €{ppm_stats['mean']:,.0f}")
        
        # Real yacht price per meter typically €50k-€2M depending on size
        if ppm_stats['mean'] > 3000000:  # Over €3M per meter is unrealistic
            issues.append("Unrealistic price per meter ratios")
        
        if ppm_stats['min'] < 10000:  # Under €10k per meter is unrealistic
            issues.append("Unrealistic low price per meter")
    
    # Check for perfect round numbers (synthetic data indicator)
    round_prices = df[price_col].apply(lambda x: x == round(x, -6) or x == round(x, -7))
    if round_prices.mean() > 0.1:  # More than 10% are round numbers
        issues.append("Suspicious number of round price numbers")
    
    return issues

def validate_locations(df):
    """Check location data"""
    print("\nValidating Locations...")
    issues = []
    
    if 'location' in df.columns:
        locations = df['location'].value_counts()
        print(f"Unique locations: {len(locations)}")
        print("Top locations:")
        for loc, count in locations.head(10).items():
            print(f"  {loc}: {count} yachts")
        
        # Real yacht market locations
        real_locations = {
            'Monaco', 'French Riviera', 'Italian Riviera', 'Spanish Coast',
            'Dubai', 'Singapore', 'Caribbean', 'Greek Islands',
            'Croatia', 'Sardinia', 'South of Florida', 'Fort Lauderdale'
        }
        
        dataset_locations = set(locations.index)
        unknown_locations = dataset_locations - real_locations
        
        if unknown_locations:
            print(f"⚠️  Unknown locations: {list(unknown_locations)}")
            issues.append(f"Suspicious/unknown locations: {list(unknown_locations)}")
    
    return issues

def validate_derived_features(df):
    """Check engineered features for synthetic patterns"""
    print("\nValidating Engineered Features...")
    issues = []
    
    # Check for perfect correlations (synthetic indicator)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    perfect_correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.999:  # Near-perfect correlation
                perfect_correlations.append((correlation_matrix.columns[i], correlation_matrix.columns[j], corr_val))
    
        if perfect_correlations:
            issues.append(f"Suspicious perfect correlations: {perfect_correlations[:5]}")
            print(f"Found {len(perfect_correlations)} near-perfect correlations")
    
    # Check for unrealistic financial metrics
    if 'annual_maintenance_cost' in df.columns and 'price_eur' in df.columns:
        maintenance_ratio = df['annual_maintenance_cost'] / df['price_eur']
        print(f"Maintenance cost ratio: {maintenance_ratio.mean():.4f} (typical: 0.03-0.10)")
        
        if maintenance_ratio.mean() > 0.15:  # Over 15% maintenance cost is high
            issues.append("Unrealistic maintenance cost ratios")
    
    # Check ROI calculations
    if 'roi_years' in df.columns:
        roi_stats = df['roi_years'].describe()
        print(f"ROI years: {roi_stats['min']:.1f} - {roi_stats['max']:.1f}")
        
        if roi_stats['min'] < 1:  # ROI under 1 year is unrealistic
            issues.append("Unrealistic ROI timeframes")
    
    return issues

def check_synthetic_patterns(df):
    """Check for common synthetic data patterns"""
    print("\nChecking for Synthetic Data Patterns...")
    issues = []
    
    # Check for perfect uniform distributions
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    uniform_distributions = []
    for col in numeric_cols[:10]:  # Check first 10 numeric columns
        if df[col].nunique() == len(df):  # All unique values
            if df[col].dtype in ['int64', 'float64']:
                # Check if values are too evenly distributed
                std_ratio = df[col].std() / df[col].mean()
                if std_ratio > 0.5:  # High variability relative to mean
                    uniform_distributions.append(col)
    
    if uniform_distributions:
        issues.append(f"Suspicious uniform distributions: {uniform_distributions}")
    
    # Check for unrealistic precision in engineered values
    precision_cols = ['efficiency_score', 'charter_demand_score', 'builder_reputation_score']
    for col in precision_cols:
        if col in df.columns:
            # Real scores typically have some variability
            unique_scores = df[col].nunique()
            if unique_scores > 900:  # Too many unique values for a "score"
                issues.append(f"Too many unique values in score column: {col}")
    
    return issues

def generate_validation_report(all_issues):
    """Generate comprehensive validation report"""
    report = f"""
# Yacht Data Validation Report
## Spot-Check for Real vs Synthetic Data

### 🔍 Executive Summary
**Validation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset Size**: 1000 yacht records
**Validation Scope**: Complete data authenticity assessment

### ⚠️ Issues Found
"""
    
    if all_issues:
        for category, issues in all_issues.items():
            if issues:
                report += f"\n#### {category}\n"
                for issue in issues:
                    report += f"- ❌ {issue}\n"
    else:
        report += "✅ **No major issues detected** - Data appears authentic\n"
    
    report += f"""
### 🔍 Validation Categories Checked

1. **Yacht ID Patterns**: Sequential ID analysis, uniqueness checks
2. **Builder Validation**: Real yacht manufacturers verification  
3. **Specification Checks**: Length, beam, GT, year validations
4. **Price Analysis**: Price distributions and ratios
5. **Location Verification**: Real yacht market locations
6. **Derived Features**: Engineered value realism
7. **Synthetic Patterns**: Uniform distributions, precision analysis

### 📊 Assessment Methodology

**Real Data Indicators** ✅:
- Varied yacht specifications
- Irregular price patterns
- Known yacht builders
- Realistic market locations
- Natural data variability

**Synthetic Data Indicators** ❌:
- Perfect sequential patterns
- Unrealistic specifications
- Unknown builders/locations
- Perfect correlations
- Uniform distributions

### 🎯 Conclusion
"""
    
    total_issues = sum(len(issues) for issues in all_issues.values())
    
    if total_issues == 0:
        report += "✅ **DATA APPEARS AUTHENTIC** - No synthetic indicators found"
    elif total_issues <= 3:
        report += "⚠️ **MINOR CONCERNS** - Some areas need verification"
    elif total_issues <= 7:
        report += "❌ **MODERATE CONCERNS** - Several suspicious patterns detected"
    else:
        report += "🚨 **MAJOR CONCERNS** - Strong synthetic data indicators"
    
    report += f"""

### 🔍 Recommendations
- Verify builders with yacht industry sources
- Cross-check specifications with real yacht listings
- Validate locations against yacht market data
- Review price distributions with market benchmarks
- Consider source verification for suspicious records

---
*Validation Complete - {datetime.now().strftime('%Y-%m-%d')}*
"""
    
    with open('yacht_data_validation_report.md', 'w') as f:
        f.write(report)
    
    return report

def main():
    """Main validation pipeline"""
    print("Starting Yacht Data Validation - Real vs Synthetic Detection\n")
    
    # Load data
    df = load_and_analyze_data()
    
    # Run all validation checks
    all_issues = {}
    
    all_issues['Yacht IDs'] = validate_yacht_ids(df)
    all_issues['Builders'] = validate_builders(df)
    all_issues['Specifications'] = validate_specifications(df)
    all_issues['Prices'] = validate_prices(df)
    all_issues['Locations'] = validate_locations(df)
    all_issues['Derived Features'] = validate_derived_features(df)
    all_issues['Synthetic Patterns'] = check_synthetic_patterns(df)
    
    # Generate report
    report = generate_validation_report(all_issues)
    
    # Print summary
    total_issues = sum(len(issues) for issues in all_issues.values())
    print(f"\nValidation Summary:")
    print(f"Total Issues Found: {total_issues}")
    
    for category, issues in all_issues.items():
        if issues:
            print(f"  {category}: {len(issues)} issues")
            for issue in issues[:2]:  # Show first 2
                print(f"    - {issue}")
    
    print(f"\nDetailed report saved: 'yacht_data_validation_report.md'")
    
    return all_issues, report

if __name__ == "__main__":
    issues, report = main()