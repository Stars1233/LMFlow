import unittest

from lmflow.pipeline.inferencer import ToolInferencer

CODE_1 = 'print("hello world")'
RES_1 = "hello world\n"
CODE_2 = "b=a+1\nprint(b)"
RES_2 = """Traceback (most recent call last):
  File "<string>", line 1, in <module>
NameError: name 'a' is not defined
"""


class ToolInferencerTest(unittest.TestCase):
    def setUp(self):
        # code_exec does not use model state; bypass model initialization so this
        # remains a fast, offline unit test.
        self.toolinf = object.__new__(ToolInferencer)

    def test_code_exec_1(self, code=CODE_1, expected_output=RES_1):
        toolinf_res = self.toolinf.code_exec(code)
        self.assertEqual(toolinf_res, expected_output)

    def test_code_exec_2(self, code=CODE_2):
        toolinf_res = self.toolinf.code_exec(code)
        self.assertNotEqual(toolinf_res.returncode, 0)
