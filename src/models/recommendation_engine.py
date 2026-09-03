"""
Recommendation engine for the RealEstateAI project.

This module provides content-based property recommendations using
user preferences and a transparent scoring system.
"""

from pathlib import Path

import pandas as pd


class RecommendationEngine:
    """
    Content-based real estate recommendation engine.

    The engine:
    1. Applies hard constraints.
    2. Calculates preference scores.
    3. Ranks matching properties.
    4. Returns the highest-scoring properties.
    """

    REQUIRED_COLUMNS = {
        "ListingID",
        "City",
        "Neighborhood",
        "PropertyType",
        "Condition",
        "Bedrooms",
        "Bathrooms",
        "SqFt",
        "LotSize_sqft",
        "Age_years",
        "GarageSpaces",
        "ListPrice",
        "PricePerSqFt",
        "DaysOnMarket",
        "ListingDate",
    }

    def __init__(self, data_path: str | Path):
        """
        Load the housing listings dataset.

        Parameters
        ----------
        data_path : str | Path
            Path to housing_market.csv.
        """

        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Housing listings dataset not found: {self.data_path}"
            )

        self.data = pd.read_csv(self.data_path)

        self._validate_dataset()

    def _validate_dataset(self) -> None:
        """Validate the dataset schema."""

        missing_columns = self.REQUIRED_COLUMNS - set(self.data.columns)

        if missing_columns:
            raise ValueError(
                "Dataset is missing required columns: "
                f"{sorted(missing_columns)}"
            )

    @staticmethod
    def _normalize_text(value: str) -> str:
        """Normalize text for case-insensitive comparisons."""

        return str(value).strip().lower()

    def _calculate_scores(
        self,
        candidates: pd.DataFrame,
        city: str | None = None,
        neighborhood: str | None = None,
        property_type: str | None = None,
        condition: str | None = None,
        min_bedrooms: int | None = None,
        min_bathrooms: float | None = None,
        max_price: float | None = None,
        min_sqft: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate recommendation scores.

        Maximum base preference score:

        City           = 20 points
        Neighborhood   = 25 points
        Property type  = 20 points
        Condition      = 10 points
        Bedrooms       = 10 points
        Bathrooms      = 5 points
        Price          = 5 points
        SqFt           = 5 points

        Maximum = 100 points.
        """

        candidates = candidates.copy()

        candidates["RecommendationScore"] = 0.0

        # -------------------------------------------------
        # Categorical preferences
        # -------------------------------------------------

        if city:
            city_normalized = self._normalize_text(city)

            candidates.loc[
                candidates["City"].map(self._normalize_text)
                == city_normalized,
                "RecommendationScore",
            ] += 20.0

        if neighborhood:
            neighborhood_normalized = self._normalize_text(neighborhood)

            candidates.loc[
                candidates["Neighborhood"].map(self._normalize_text)
                == neighborhood_normalized,
                "RecommendationScore",
            ] += 25.0

        if property_type:
            property_type_normalized = self._normalize_text(property_type)

            candidates.loc[
                candidates["PropertyType"].map(self._normalize_text)
                == property_type_normalized,
                "RecommendationScore",
            ] += 20.0

        if condition:
            condition_normalized = self._normalize_text(condition)

            candidates.loc[
                candidates["Condition"].map(self._normalize_text)
                == condition_normalized,
                "RecommendationScore",
            ] += 10.0

        # -------------------------------------------------
        # Bedroom preference
        # -------------------------------------------------

        if min_bedrooms is not None:
            bedroom_match = (
                candidates["Bedrooms"] >= min_bedrooms
            )

            candidates.loc[
                bedroom_match,
                "RecommendationScore",
            ] += 10.0

        # -------------------------------------------------
        # Bathroom preference
        # -------------------------------------------------

        if min_bathrooms is not None:
            bathroom_match = (
                candidates["Bathrooms"] >= min_bathrooms
            )

            candidates.loc[
                bathroom_match,
                "RecommendationScore",
            ] += 5.0

        # -------------------------------------------------
        # Price preference
        # -------------------------------------------------

        if max_price is not None:
            price_ratio = (
                max_price - candidates["ListPrice"]
            ) / max_price

            price_ratio = price_ratio.clip(lower=0.0, upper=1.0)

            candidates["RecommendationScore"] += (
                price_ratio * 5.0
            )

        # -------------------------------------------------
        # Square footage preference
        # -------------------------------------------------

        if min_sqft is not None:
            sqft_match = candidates["SqFt"] >= min_sqft

            candidates.loc[
                sqft_match,
                "RecommendationScore",
            ] += 5.0

        return candidates

    def recommend(
        self,
        city: str | None = None,
        property_type: str | None = None,
        condition: str | None = None,
        min_bedrooms: int | None = None,
        min_bathrooms: float | None = None,
        max_price: float | None = None,
        min_sqft: int | None = None,
        max_sqft: int | None = None,
        neighborhood: str | None = None,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Generate ranked property recommendations.

        Parameters
        ----------
        city : str, optional
            Preferred city.

        property_type : str, optional
            Preferred property type.

        condition : str, optional
            Preferred property condition.

        min_bedrooms : int, optional
            Minimum bedrooms.

        min_bathrooms : float, optional
            Minimum bathrooms.

        max_price : float, optional
            Maximum listing price.

        min_sqft : int, optional
            Minimum square footage.

        max_sqft : int, optional
            Maximum square footage.

        neighborhood : str, optional
            Preferred neighborhood.

        top_n : int, default=5
            Number of recommendations.

        Returns
        -------
        pd.DataFrame
            Ranked recommendations.
        """

        if top_n <= 0:
            raise ValueError("top_n must be greater than 0.")

        candidates = self.data.copy()

        # -------------------------------------------------
        # Hard filters
        # -------------------------------------------------

        if city:
            city_normalized = self._normalize_text(city)

            candidates = candidates[
                candidates["City"].map(self._normalize_text)
                == city_normalized
            ]

        if neighborhood:
            neighborhood_normalized = self._normalize_text(
                neighborhood
            )

            candidates = candidates[
                candidates["Neighborhood"].map(self._normalize_text)
                == neighborhood_normalized
            ]

        if property_type:
            property_type_normalized = self._normalize_text(
                property_type
            )

            candidates = candidates[
                candidates["PropertyType"].map(self._normalize_text)
                == property_type_normalized
            ]

        if min_bedrooms is not None:
            candidates = candidates[
                candidates["Bedrooms"] >= min_bedrooms
            ]

        if min_bathrooms is not None:
            candidates = candidates[
                candidates["Bathrooms"] >= min_bathrooms
            ]

        if max_price is not None:
            candidates = candidates[
                candidates["ListPrice"] <= max_price
            ]

        if min_sqft is not None:
            candidates = candidates[
                candidates["SqFt"] >= min_sqft
            ]

        if max_sqft is not None:
            candidates = candidates[
                candidates["SqFt"] <= max_sqft
            ]

        # -------------------------------------------------
        # Handle no matches
        # -------------------------------------------------

        if candidates.empty:
            return pd.DataFrame(
                columns=[
                    "ListingID",
                    "City",
                    "Neighborhood",
                    "PropertyType",
                    "Condition",
                    "Bedrooms",
                    "Bathrooms",
                    "SqFt",
                    "LotSize_sqft",
                    "Age_years",
                    "GarageSpaces",
                    "ListPrice",
                    "PricePerSqFt",
                    "DaysOnMarket",
                    "ListingDate",
                    "RecommendationScore",
                ]
            )

        # -------------------------------------------------
        # Calculate recommendation scores
        # -------------------------------------------------

        candidates = self._calculate_scores(
            candidates=candidates,
            city=city,
            neighborhood=neighborhood,
            property_type=property_type,
            condition=condition,
            min_bedrooms=min_bedrooms,
            min_bathrooms=min_bathrooms,
            max_price=max_price,
            min_sqft=min_sqft,
        )

        # -------------------------------------------------
        # Rank properties
        # -------------------------------------------------

        candidates = candidates.sort_values(
            by=[
                "RecommendationScore",
                "ListPrice",
            ],
            ascending=[
                False,
                True,
            ],
        )

        return candidates.head(top_n).reset_index(drop=True)

    def get_listing(self, listing_id: str) -> pd.Series | None:
        """
        Retrieve a listing using its ListingID.
        """

        matches = self.data[
            self.data["ListingID"].astype(str)
            == str(listing_id)
        ]

        if matches.empty:
            return None

        return matches.iloc[0]