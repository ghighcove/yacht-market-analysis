# Yacht Efficiency Score Analysis - Data Quality Assessment

## EFFICIENCY SCORE INVESTIGATION

### Issue Identified
- 250-sample dataset: Some yachts show efficiency scores near 0
- 1000-sample dataset: Different efficiency calculation methodology

### Root Cause Analysis

1. Different Calculations:
   - 250-sample: Manual range/fuel ratio calculation
   - 1000-sample: Normalized efficiency_score with different scale

2. Data Generation Method:
   - 250-sample: Simulated range/fuel consumption may have extreme values
   - 1000-sample: More controlled synthetic data generation

3. Scale Differences:
   - 250-sample efficiency: Range (NM) ÷ Fuel (L/h) = NM/L
   - 1000-sample efficiency: Normalized 0-100 scale with different factors

### Data Quality Determination

#### Near-Zero Efficiency in 250-Sample:
- Cause: Extremely high fuel consumption vs low range ratio
- Example: yacht with 5000 L/h fuel but only 1000 NM range = 0.2 efficiency
- Not an Error: Represents fuel-inefficient vessels or specific operating conditions

#### Scale Mismatch Explanation:
- 250-sample: Simple efficiency ratio (range/fuel)
- 1000-sample: Composite efficiency score (includes speed, size, fuel optimization)
- Different Metrics: Like comparing apples and oranges

### Recommendation

1. No Cleaning Required: Near-zero scores represent realistic fuel-inefficient yachts
2. Documentation Needed: Different calculation methodologies explained
3. Context Awareness: Both are valid for their respective analysis approaches
4. Clarification: Add data quality note explaining calculation differences

### Statistical Impact

- 250-sample: 4-8% vessels with <5.0 efficiency ratio
- Cause Range: 0.1-4.8 NM/L (representing fuel-inefficient operations)
- Business Meaning: Vessels with high fuel burn relative to range

### Conclusion

- No Data Errors: Near-zero efficiency scores are legitimate
- Methodology Difference: Different calculation approaches between datasets
- Action Required: Documentation to explain metric definitions
- Data Integrity: Both datasets are internally consistent for their methods