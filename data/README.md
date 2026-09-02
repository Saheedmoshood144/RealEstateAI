# Dataset Documentation

This project uses two datasets for different purposes in the real-estate
intelligence pipeline.

## 1. California Housing Dataset

File:

`data/raw/california_housing.csv`

Purpose:

Used for machine-learning experiments involving California housing
characteristics and property-value prediction.

Schema verification:

- Rows: 20,640
- Columns: 9

The dataset schema must be treated as authoritative when building the
preprocessing and modeling pipeline.

## 2. Housing Listings Dataset

File:

`data/raw/housing_market.csv`

Rows: 1,500

Columns: 15

Columns:

- ListingID
- City
- Neighborhood
- PropertyType
- Condition
- Bedrooms
- Bathrooms
- SqFt
- LotSize_sqft
- Age_years
- GarageSpaces
- ListPrice
- PricePerSqFt
- DaysOnMarket
- ListingDate

Validation performed:

- No missing values
- No duplicate ListingID values
- 5 unique cities
- 17 unique neighborhoods
- 4 property types
- 4 property conditions

## Licensing and Provenance

The project will only redistribute datasets for which the source license
explicitly permits redistribution.

The listings dataset must not be represented as CC0 unless its original
source provides explicit evidence that it is released under CC0.

No Zillow, Redfin, Realtor.com, or other proprietary listing data is
intentionally scraped or redistributed by this project.