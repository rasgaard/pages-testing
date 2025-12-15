---
title: Use-case for Prompting Whisper in Transformers.js
date: 2024-08-29
layout: default
tags:
  - post
  - code
---
# {{page.title}}
At Laerdal we are always trying to maximize learning outcome of our products by providing an experience that closely resembles a real medical emergency situation. One way to improve the learning outcome is to use voice recognition as an interface to our products. Often the user is instructed to say something in a stage of the learning experience. This could be "Get an AED" or "Check for breathing". We need to be sure that the user actually says the correct phrase in order to proceed.

  

Transcription quality varies a lot between services. A particular problem is that the user might say the right thing but the transcription is incorrect, resulting in a poor user experience. This was very often the case in jargon such as "AED" which the transcription service would often hear as "80", "AD", etc.

  

I found out that it is possible to [prompt Whisper](https://cookbook.openai.com/examples/whisper_prompting_guide) in order to guide the text generation. This could be particularly useful in our case where we knew that the audio should contain a particular phrase or word.

  

This was very easy to try out and validate using the [Transformers library](https://github.com/huggingface/transformers/) due to the `.generate` method that allows for passing in a prompt as shown here:

```python
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
whisper = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny.en")

  

prompt = torch.tensor(processor.get_prompt_ids("get an AED"))
audio_input = ... # audio file to be transcribed

input_features = processor(
	audio_input, sampling_rate=16_000, return_tensors="pt"
).input_features

predicted_ids = whisper.generate(input_features, prompt_ids=prompt)
processor.batch_decode(
	predicted_ids,
	skip_special_tokens=True,
)
```

  

The twist is that we would like to use this in an application that uses Javascript as the backend. I have previously experimented with [transformers.js](https://huggingface.co/docs/transformers.js/) but never had a reason to dig very deep.

  

As I have very limited experience using Javascript it was really nice to find that there are good [examples](https://github.com/xenova/whisper-web) of using Whisper with transformers.js. However, I quickly realized that there was no support for prompting Whisper as with the Python library.

  

In order to make this work I had to dig deeper into the transformers library to find out how the prompt was actually implemented. I found that the prompt was actually just a list of token ids using the special `<|startofprev|>`-token. This was pretty easy to implement in Javascript as well and could be passed as the `decoder_input_ids` as they are [practically the same](https://github.com/huggingface/transformers/issues/28228) as the `prompt_ids`.

  
  

```javascript

import { AutoProcessor, AutoTokenizer, WhisperForConditionalGeneration } from "@huggingface/transformers";
import { readFileSync } from 'fs';
import * as wav from 'node-wav';

  

const model = await WhisperForConditionalGeneration.from_pretrained("onnx-community/whisper-tiny.en", {
	dtype: {
		encoder_model: 'fp16',
		decoder_model_merged: 'q4',
	}
});

  

const tokenizer = await AutoTokenizer.from_pretrained("onnx-community/whisper-tiny.en");
const processor = await AutoProcessor.from_pretrained("onnx-community/whisper-tiny.en");

const decoded_audio = ... // audio file to be transcribed

const input = await processor(decoded_audio.channelData[0]);

let prompt_str = "<|startofprev|> get an AED"
let prompt_ids = tokenizer.encode(prompt_str, { add_special_tokens: false });

const output = await model.generate({ inputs: input.input_features, decoder_input_ids: [prompt_ids] });

const decodedOutput = tokenizer.decode(output[0], { skip_special_tokens: true });

const output_str = decodedOutput.slice(prompt_str.length - "<|startofprev|>".length);
```

  

These two snippets achieve exactly the same thing but some tweaking had to be done in the case of transformers.js as it isn't as feature complete as the Python library. This was a really cool learning experience that helped me understand the underlying mechanics of what the tokens actually do and how they can be used for Whisper transcription.