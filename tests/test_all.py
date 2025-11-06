"""run all available tests"""
import pytest

if __name__ == '__main__':
    pytest.cmdline.main(['tests/temperature_ladder_tests.py', 'tests/test_temperature_construction.py'])
