#!/usr/bin/env python
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""
Merge base model and lora model into a full model.
"""

import os
import sys

sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))

from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser

from lmflow.args import (
    ModelArguments,
)
from lmflow.models.auto_model import AutoModel


@dataclass
class MergeLoraArguments:
    device: str = field(
        default="cpu",
        metadata={
            "help": "device to merge model on",
        },
    )
    output_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "output merged full model path"},
    )
    local_rank: Optional[int] = field(
        default=-1,
        metadata={
            "help": "local rank for deepspeed",
        },
    )


def main():
    parser = HfArgumentParser((ModelArguments, MergeLoraArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, merge_lora_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, merge_lora_args = parser.parse_args_into_dataclasses()

    if merge_lora_args.device == "gpu":
        raise NotImplementedError("Merging LoRA weight using GPU not supported yet. Please use cpu.")

    model_args.use_lora = True
    model = AutoModel.get_model(
        model_args,
        do_train=False,
        device=merge_lora_args.device,
    )
    model.activate_model_for_inference()
    model.merge_lora_weights()
    model.save(merge_lora_args.output_model_path, save_full_model=True)


if __name__ == "__main__":
    main()
