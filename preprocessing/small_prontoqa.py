# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json

file = json.load(open("data/5hop_0shot_random.json"))
data = []

for k, v in file.items():
    example = v["test_example"]
    data.append(
        {
            "question": example["question"] + " " + example["query"],
            "steps": [
                " ".join(example["chain_of_thought"][i : i + 2])
                for i in range(0, len(example["chain_of_thought"]), 2)
            ],
            "answer": example["answer"],
        }
    )

json.dump(data[:400], open("data/prontoqa_train.json", "w"))
json.dump(data[400:420], open("data/prontoqa_valid.json", "w"))
json.dump(data[420:], open("data/prontoqa_test.json", "w"))
