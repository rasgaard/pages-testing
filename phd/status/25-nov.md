---
date: 2025-11-24
tags:
  - status
title: "Early Project Update: Model Compression and Edge ASR Directions"
month: November
year: 2025
---
# {{ title }}
As part of the industrial PhD I have to take a course aptly named "Industrial PhD Course". During this course I have had to think about *stakeholder management* a fair bit. Mapping out the stakeholders of the project and how to make sure that no one feels like they're missing out. 

I would like to try out making status updates accessible by anyone. This serves multiple purposes:
1. A place to refer stakeholders who show interest in the project.
2. Smoothen communication with supervisors from both the company and university.
3. An archive of what I have done and a place to jot down notes, papers, etc. This one is mostly for my own sake.

As I haven't done this for the first month of the project in October I will try to include Oct. activities as well here. 
## Reading, Prototyping, and Soaking Up SOTA
First two months of the project has passed. The main activities of this initial period can be summarized as scoping, reading and testing. We have a scope for what we would like to do: Compress Automatic Speech Recognition models - Whisper in this case - using pruning and quantization to understand how it affects the model. 

I have been spending a significant amount of time familiarizing myself with state-of-the-art research on model compression. Although I didn't start the project with a solid theoretical foundation on the topics I did have some clues for what resources and researchers to pay attention to. The [efficientml.ai](https://hanlab.mit.edu/courses/2024-fall-65940) lectures are fantastic to cover a lot of grounds in a fairly short amount of time.

Here are some of my favorite papers I have been reading:
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://www.semanticscholar.org/paper/LLM.int8()%3A-8-bit-Matrix-Multiplication-for-at-Dettmers-Lewis/4be7d1524edb0137599a5cc95f72844b85a52fe1)
- [A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/pdf/2306.11695)
- [SLiM: One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression](https://www.semanticscholar.org/paper/SLiM%3A-One-shot-Quantization-and-Sparsity-with-for-Mozaffari-Yazdanbakhsh/edd3d142b1886fefc68b87befb894099d00ad7ae)
- [MaskLLM: Learnable Semi-Structured Sparsity for Large Language Models](https://www.semanticscholar.org/paper/MaskLLM%3A-Learnable-Semi-Structured-Sparsity-for-Fang-Yin/60263d3ef1604274e5f095803e9c437f860c511f)
- [WhisperKit: On-device Real-time ASR with Billion-Scale Transformers](https://www.semanticscholar.org/paper/WhisperKit%3A-On-device-Real-time-ASR-with-Orhon-Okan/8ff9d940fe4c0b578dacc62b670c69cfc56b6057)
- [Llama Guard 3-1B-INT4: Compact and Efficient Safeguard for Human-AI Conversations](https://www.semanticscholar.org/paper/Llama-Guard-3-1B-INT4%3A-Compact-and-Efficient-for-Fedorov-Plawiak/bbf8d44623354a1e560d45e9950d905dd6f02bec)
- [Edge-ASR: Towards Low-Bit Quantization of Automatic Speech Recognition Models](https://www.semanticscholar.org/paper/Edge-ASR%3A-Towards-Low-Bit-Quantization-of-Automatic-Feng-Lin/45bb4cbb2c4064c848b23752ec7cb42563f6a700)

My overall goal at this point in time is to build up an excellent foundation for further research.
## Recent PyTorch Posts on Sparsity
I'm excited to see that the PyTorch blog chose to publish not just one but two posts about adding sparsity to the mix of model compression. 
* [When Quantization Isn’t Enough: Why 2:4 Sparsity Matters ](https://pytorch.org/blog/when-quantization-isnt-enough-why-24-sparsity-matters/)
* [Beyond Quantization: Bringing Sparse Inference to PyTorch](https://pytorch.org/blog/beyond-quantization-bringing-sparse-inference-to-pytorch/) 
## ASR Compression and Edge Deployment
We needed something to focus on. Otherwise I felt that there would be a major risk of wandering around topics and not producing anything noteworthy. We chose compression and edge deployment of Automatic Speech Recognition (ASR) models - specifically Whisper. Voice is a common interface for Laerdal's simulation scenarios, making it a an obvious candidate. 

Furthermore, compression of ASR models is an unexplored topic in academia. The hope is that we can get inspired by research on LLM compression and produce similar tables to those in the first [PyTorch blog post](https://pytorch.org/blog/when-quantization-isnt-enough-why-24-sparsity-matters/).

To do this successfully I need to build up the practical know-how of applying the compression methods and deployment to edge devices. I have been looking a bit into [ExecuTorch](https://executorch.ai/) which looks extremely promising.

### Laerdal's interests in offline transcriptions
To better understand one of the promising company use-cases I tagged along a trip to Stavanger where Laerdal's headquarter office is. Laerdal Medical's sister organization, Laerdal Global Health, has had incredible results providing [simulation training for underserved communities](https://laerdalglobalhealth.com/partnerships-and-programs/safer-births-research-project/). 

The finals steps of the simulation training is debriefing and documentation. We saw how tedious and error-prone the current manual workflow is when documenting the simulation. If we can somehow use AI to ease this process we could collect even better data to further develop the programme.

Access to internet is common but under vastly different conditions than what I'm used to. People are often on limited data plans, causing them to be very protective of data usage. Uploading large audio files to be transcribed is simply not an option at a lot of these sites.

If we can transcribe the audio, thereby "compressing" the audio to text, we suddenly use a lot less data to transfer the transcription instead of raw audio.

## Quantization, Sparsity, and Early WER Results
My first attempt at compressing Whisper (small, 244M parameters) was done with [TorchAO](https://docs.pytorch.org/ao/stable/index.html). As this was completely new to me I reached out on the [GPU MODE discord](https://discord.com/invite/gpumode) and got some much needed help to get started. Jesse Cai, a TorchAO maintainer, who coincidentally also wrote the PyTorch blog post, helped me out. Very much appreciated!

From the ASR literature I've found that [librispeech](https://huggingface.co/datasets/openslr/librispeech_asr) was commonly used as a benchmarking dataset. The Word Error Rates (WER) below are on the clean.other part of librispeech test set.  About 300 samples from the [People's Speech Dataset](https://huggingface.co/datasets/MLCommons/peoples_speech) was used as the calibration dataset.

| Quantization   | Sparsity               | MB   | Calibration | WER      |
| -------------- | ---------------------- | ---- | ----------- | -------- |
| -              | -                      | 1074 | -           | 4.27%    |
| Int8WeightOnly | -                      | 578  | -           | 4.237%   |
| -              | Wanda 50% unstructured | 1074 | -           | 101.204% |
| -              | Wanda 50% unstructured | 1074 | 1 epoch     | 17.748%  |
| -              | Wanda 50% unstructured | 1074 | 2 epochs    | 17.632%  |

Next obvious step is to combine the methods and run the combination sweep on more Whisper variants. 

But I'll put that on the shelf for a bit. I want to make sure that I dedicate some time to the edge deployment branch of the project as well. Now that I have some rudimentary understanding of model compression I would like to go through the same process but for edge deployment.

## The coming months

Looking at december and january I will be occupied with teaching assistant duties, development of course tooling and hopefully a bit of holiday.

| Week | Activity                         |
| ---- | -------------------------------- |
| 49   | EurIPS                           |
| 50   | MLOps development                |
| 51   | Project work & MLOps development |
| 52   | Holidays                         |
| 1    | Holidays                         |
| 2    | MLOps Teaching Asisstant         |
| 3    | MLOps Teaching Asisstant         |
| 4    | MLOps Teaching Asisstant         |
