"""run all available tests"""

import pytest

if __name__ == '__main__':
    pytest.cmdline.main(['tests/test_temperature_ladder.py', 'tests/test_temperature_construction.py'])
