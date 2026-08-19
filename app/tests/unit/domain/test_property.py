import pytest

from app.domain.entities.property import Property

VALID_KWARGS = dict(
    bedrooms=3, bathrooms=2.0, sqft_living=1800, sqft_lot=5000, floors=1.0, waterfront=0,
    view=0, condition=3, grade=7, sqft_above=1800, sqft_basement=0, yr_built=1990,
    yr_renovated=0, zipcode=98101, lat=47.6, long=-122.3, sqft_living15=1800, sqft_lot15=5000,
)


def test_valid_property_constructs():
    prop = Property(**VALID_KWARGS)
    assert prop.grade == 7


@pytest.mark.parametrize("field,value", [
    ("bathrooms", 11), ("floors", 6), ("grade", 14), ("grade", 0), ("condition", 6),
    ("waterfront", 2), ("view", 5), ("sqft_living", -1), ("yr_built", 1500),
])
def test_invalid_property_raises(field, value):
    kwargs = {**VALID_KWARGS, field: value}
    with pytest.raises(ValueError):
        Property(**kwargs)
