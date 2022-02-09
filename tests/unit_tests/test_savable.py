import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)

from base.savable import Savable 

import unittest

class TestSavable(unittest.TestCase):

    def test_get_pickleable_attributes(self):
        savable_obj = Savable()
        savable_obj.testAttribute1 = 10
        savable_obj.testAttribute2 = "test"

        #Arrange
        pickleable_attributes  = savable_obj.get_pickleable_attributes()

        #Act
        pickleable_attr_correct = verify_pickleable_attributes(pickleable_attributes)

        #Assert
        self.assertEqual(True, pickleable_attr_correct)

    def test_get_pickleable_dict(self):
        savable_obj = Savable()
        savable_obj.testAttribute1 = 10
        savable_obj.testAttribute2 = "test"

        #Arrange
        pickleable_dict  = savable_obj.get_pickleable_dict()

        #Act
        pickleable_dict_correct = verify_pickleable_dict(pickleable_dict)

        #Assert
        self.assertEqual(True, pickleable_dict_correct)


'''
Auxiliary verification functions for savable tester
'''

def verify_pickleable_attributes(pickleable_attributes):
    if "testAttribute1" in pickleable_attributes and "testAttribute2" in pickleable_attributes:
        return True
    return False

def verify_pickleable_dict(pickleable_dict):
    if pickleable_dict['testAttribute1'] == 10 and pickleable_dict['testAttribute2'] == "test":
        return True
    return False


if __name__ == '__main__': 
    unittest.main()
