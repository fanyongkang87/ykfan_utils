from unittest import TestCase
from ykfan_utils.common_utils.example import ex_add


class TestExample(TestCase):
    def test_result(self):
        res = ex_add()
        self.assertEqual(res, 3, msg='ex add result should be 3.')
