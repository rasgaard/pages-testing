---
title: Fine-tuning Whisper-tiny for Danish (for free)
date: 2025-05-25
tags:
  - post
  - code
---
# {{title}}
## Appreciation for Small Models and Open Datasets

I have a long-standing appreciation for small yet capable AI models. Being lightweight yet highly performant is so impressive to me. Constraints on model size go a little bit against the grain for state-of-the-art performance as we are constantly witnessing scale being the predominant factor for success. I think this contributes to my fascination. Small models have a certain "underdog" quality to them. Additionally, I believe that putting such constraints on AI systems can result in more interesting solutions.

  

It can be hard to scratch the model training itch without breaking the bank if you are just a hobbyist with limited computational resources. That's the case for larger models, at least. It is entirely possible to use free resources on Google Colab or Kaggle to train smaller models without ever picking up your credit card. Realizing this got me excited, as it opens the door for hobbyists and tinkerers to explore the space, investing only their time and attention.

  

The models are only half of the story. Data needs to be accessible too. Projects like [CoRal](https://alexandra.dk/coral/) by the Alexandra Institute have done an amazing job at collecting high-quality datasets that are available for everyone. It is also highly beneficial to everyone. Having open datasets puts them under public scrutiny and ensures their quality, ultimately leading to better AI systems.

  

## Whisper-tiny.da: Putting the pieces together

I have been looking for an excuse to play around more with Whisper lately. The `tiny` variant with its 40M parameter size allows it to run smoothly on mobile phones and perform real-time transcription in the browser using [Transformers.js](https://huggingface.co/spaces/Xenova/whisper-webgpu).

  

The question that I wanted to answer was: "What performance can I squeeze out of Whisper-tiny if trained on high-quality Danish speech?". I wanted to know where on this [Word/Character Error Rate table](https://huggingface.co/CoRal-project/roest-wav2vec2-315m-v1#evaluation-results) it would land.

  

Luckily for me, a lot of the hard work was already done by other people and packaged up nicely. I read through a [tutorial post](https://huggingface.co/blog/fine-tune-whisper) and copied relevant snippets over to a Kaggle notebook. Kaggle offers 30 GPU hours per week which turned out to be plenty for this small project. I later found that the team behind CoRal had put together [training and evaluation scripts](https://github.com/alexandrainst/coral) for this exact purpose, making the whole process incredibly easy. I would have cheated myself of the (painful) experiences and lessons of debugging a whole mess of special tokens that got tangled up in the process of stitching together code had I used their code.

  

After having all the pieces in place — streaming dataset, data collator, training arguments, evaluation code — it was finally time to press Execute Call on `trainer.train()` and hopefully watch the Word Error Rate (WER) go down.

## How did it do then?

Here are the results:

  

- Character Error Rate (CER): 15.93%

- Word Error Rate (WER): 34.30%

  

This would indicate that it performs similarly as models that are **40x** the size. Wow, that's amazing! ...right?

  

Well, let's not get ahead of ourselves. Let's try to look at an actual transcription.

  

To try it out I'll start up a temporary Jupyter server with uv:

```bash
> uv run --with ipython --with jupyter --with transformers --with torch
```

and copy the Jupyter server URL into VS Code. I find it to be a nice way to create minimal throwaway/temporary environments for testing for one-off work.

  

I like to test on samples that are not in the original dataset — not even the test set — to set expectations for actual usage. The CoRal project has another [dataset for training Text-to-Speech](https://huggingface.co/datasets/alexandrainst/coral-tts) models.

  

My favorite sample from the CoRal-TTS dataset has the voice actor say "Jeg fucking elsker tebirkes[^1]".

  

Let's first benchmark against the original [`whisper-tiny`](https://huggingface.co/openai/whisper-tiny) model from OpenAI

```python
from transformers import pipeline

audio_link = "https://..."
pipe = pipeline(task="automatic-speech-recognition",
model="openai/whisper-tiny")
pipe(audio_link)
>>> {'text': ' Ja, fucking, ells godt, tæt biogis.'}
```

  

and then our CoRal fine-tuned [whisper-tiny.da](https://huggingface.co/rasgaard/whisper-tiny.da/)

  

```python
pipe = pipeline(task="automatic-speech-recognition",
model="rasgaard/whisper-tiny.da")
pipe(audio_link)
>>> {'text': 'erforking elsker 10 birkes'}
```

  

And finally comparing it to [`syvai/hviske2`](https://huggingface.co/syvai/hviske-v2):

```python
pipe = pipeline(task="automatic-speech-recognition",
model="syvai/hviske-v2")
pipe(audio_link)
>>> {'text': 'jeg fucking elsker tebirkes'}
```

...and we can see that there's still some way to go for state-of-the-art performance.

  

But that's not really what I care about here. I care about *best-in-class* performance. While it is apparent that there is a long way to go for any kind of *good* or even *slightly useful* performance I think it is really interesting to pursue.

  

## What's next?

There are a few things that I see as immediate next steps:

  

- More data. I'd love to try out even training on more data to see if generalization across datasets could be improved. A dataset that I can immediately try is [Common Voice by Mozilla](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0/). I'll get around to that someday soon.

  

- Another thing I have considered is that the model doesn't produce any punctuation and only uses lower-case. This is due to the way the CoRal dataset was collected. I have thought about doing some LLM-processing and inserting punctuation and casing into the text in CoRal.

  

- Lastly, I'd like to create something useful with the model. Transcription services are highly useful and can even be quite a good [business case](https://goodtape.io/). Having decent transcriptions without the need to send the audio off to a server could enable more use cases that are constrained on privacy. It also allows transcriptions to be done directly on low-power devices such as phones.

  

I hope to continue this project and extend it into something bigger. It has been fun to have a hobby project where I tried to work within the limitations of what is possible to do when you are GPU-poor and rely on free/openly available services and datasets.

  

Reach out to me on [LinkedIn](https://www.linkedin.com/in/rasgaard/) or [Bluesky](https://bsky.app/profile/rasgaard.com) if you also get excited about the possibilities of smaller models!

  

[^1]: Translates to "I fucking love tebirkes". Tebirkes is a Danish poppy seed pastry. It has the transcription id 206 in the dataset.