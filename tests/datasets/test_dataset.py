#!/bin/env/python3
"""A one-line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""

import json
import os
import tempfile
import unittest

from lmflow.args import DatasetArguments
from lmflow.datasets.dataset import Dataset


class DatasetTest(unittest.TestCase):
    def test_init(self):
        json_obj = {
            "type": "text2text",
            "instances": [
                {"input": "INPUT 1", "output": "OUTPUT 1"},
                {"input": "INPUT 2", "output": "OUTPUT 2"},
            ],
        }
        with tempfile.TemporaryDirectory() as dataset_dir:
            with open(os.path.join(dataset_dir, "train.json"), "w", encoding="utf-8") as fout:
                json.dump(json_obj, fout)

            data_args = DatasetArguments(dataset_path=dataset_dir)
            dataset = Dataset(data_args, backend="huggingface")
            hf_dataset = dataset.get_backend_dataset()

            self.assertEqual(len(hf_dataset), len(json_obj["instances"]))
            for expected, actual in zip(json_obj["instances"], hf_dataset):
                self.assertEqual(expected, actual)

    def test_create_from_dict(self):
        data_dict = {
            "type": "text2text",
            "instances": [
                {"input": "INPUT 1", "output": "OUTPUT 1"},
                {"input": "INPUT 2", "output": "OUTPUT 2"},
            ],
        }
        dataset = Dataset.create_from_dict(data_dict)
        self.assertEqual(dataset.to_dict(), data_dict)

    def test_create_from_dict_bad_type(self):
        data_dict = {
            "type": "non-supported",
            "instances": [
                {"input": "INPUT 1", "output": "OUTPUT 1"},
                {"input": "INPUT 2", "output": "OUTPUT 2"},
            ],
        }
        with self.assertRaises(ValueError):
            dataset = Dataset.create_from_dict(data_dict)
