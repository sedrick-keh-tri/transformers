<!--Copyright 2024 OpenLM and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# OpenLM

## Overview

The HF class allows running models trained using the [OpenLM](https://github.com/mlfoundations/open_lm/) codebase. 

It uses OpenLM as a dependency with `pip install git+https://github.com/mlfoundations/open_lm.git`.

Read more about OpenLM [in the release blogpost](https://laion.ai/blog/open-lm/)

(Note: This currently does not support some of the complex features of OpenLM such as Mixture-of-Experts.)


## Sample Usage
OpenLM relies on xFormers for its attention implementation. Hence, it requires a GPU in order to work properly. 
```python
>>> from transformers import AutoTokenizer, OpenLMForCausalLM

>>> model = OpenLMForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS).to("cuda")
>>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

>>> prompt = "Hey, are you conscious? Can you talk to me?"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(inputs.input_ids.to("cuda"), max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
```


## OpenLMConfig

[[autodoc]] OpenLMConfig
    - all

## OpenLMModel

[[autodoc]] OpenLMModel
    - forward

## OpenLMForCausalLM

[[autodoc]] OpenLMForCausalLM
    - forward