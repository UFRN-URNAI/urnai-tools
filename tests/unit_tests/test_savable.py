import inspect
import os
import sys

from urnai.base.savable import Savable

def test_get_pickleable_attributes():
    savable_obj = Savable()
    savable_obj.testAttribute1 = 10
    savable_obj.testAttribute2 = 'test'

    # Arrange
    pickleable_attributes = savable_obj.get_pickleable_attributes()

    # Assert
    assert 'testAttribute1' in pickleable_attributes
    assert 'testAttribute2' in pickleable_attributes

def test_get_pickleable_dict():
    savable_obj = Savable()
    savable_obj.testAttribute1 = 10
    savable_obj.testAttribute2 = 'test'

    # Arrange
    pickleable_dict = savable_obj.get_pickleable_dict()

    # Assert
    assert pickleable_dict['testAttribute1'] == 10
    assert pickleable_dict['testAttribute2'] == 'test'
