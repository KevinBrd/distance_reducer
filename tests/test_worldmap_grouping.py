import numpy as np
import pandas as pd
from docplex.mp.model import Model

import pytest

from distance_reducer import get_cartesian, cartesian_distance, add_cartesian_coordinates, create_distance_table, \
    create_binary_var_dict, create_continuous_var_dict, distance_between_elements


# Test for get_cartesian function
def test_get_cartesian():
    lat = 40
    long = -75
    expected_result = (784.8989691760897, -2929.2828317736057)
    assert get_cartesian(lat, long) == pytest.approx(expected_result)


# Test for cartesian_distance function
def test_cartesian_distance():
    point1 = (0, 0)
    point2 = (3, 4)
    assert cartesian_distance(point1, point2) == pytest.approx(5)


# Test for add_cartesian_coordinates function
def test_add_cartesian_coordinates():
    # Create a sample DataFrame
    data = pd.DataFrame({'Lat': [40, 34.05], 'Long': [-75, -118.2]})
    result = add_cartesian_coordinates(data)
    # Expected result after adding Cartesian coordinates
    expected_result = pd.DataFrame({'Lat': [40, 34.05],
                                    'Long': [-75, -118.2],
                                    'X': [784.8989691760897, -1549.9952617936228],
                                    'Y': [-2929.2828317736057, -2890.728946934687]})
    pd.testing.assert_frame_equal(result, expected_result)


# Test for create_distance_table function
def test_create_distance_table():
    data = pd.DataFrame({'Name': ['A', 'B', 'C'],
                         'X': [0, 1, 2],
                         'Y': [0, 1, 2]})
    expected_result = np.array([[0, np.sqrt(2), np.sqrt(8)],
                                [np.sqrt(2), 0, np.sqrt(2)],
                                [np.sqrt(8), np.sqrt(2), 0]])
    result = create_distance_table(data)
    np.testing.assert_allclose(result, expected_result)


# Test for create_binary_var_dict function
def test_create_binary_var_dict():
    model = Model()
    number_elements = 5
    number_of_groups = 3
    result = create_binary_var_dict(model, number_elements, number_of_groups)
    assert len(result) == number_elements * number_of_groups


# Test for create_continuous_var_dict function
def test_create_continuous_var_dict():
    model = Model()
    number_elements = 5
    result = create_continuous_var_dict(model, number_elements)
    assert len(result) == number_elements ** 2


# Test for distance_between_elements function
def test_distance_between_elements():
    data = pd.DataFrame({'Name': ['A', 'B'],
                         'Lat': [40.7128, 34.0522],
                         'Long': [-74.0060, -118.2437]})
    elemA = 'A'
    elemB = 'B'
    expected_result = 3943  # Approximate distance in kilometers
    result = distance_between_elements(data, elemA, elemB)
    assert result == pytest.approx(expected_result, 1)


# Run tests
if __name__ == "__main__":
    pytest.main()
