---
title: Getting Started with Transformers.js
date: 2025-05-29
layout: default
tags:
  - post
  - code
---
# {{page.title}}
Integrating machine learning into applications often involves creating APIs, models on servers, and complex deployment pipelines. But what if you could run AI models directly in your users' browsers? That's where Transformers.js comes in - bringing the power of Hugging Face models to JavaScript with minimal setup.

I recently appeared on [Verbos Podcast](https://verbospodcast.dk/) as a guest where we talked about serving machine learning models locally in the browser. During the episode the host, [Kasper Junge](https://www.linkedin.com/in/kasper-juunge), asked something along the lines of "So, how do you get started? How and where do you actually run something like this?"

In this post, I'll try to boil down the few steps that are needed to getting started with Transformers.js. I'll keep the example to a barebones implementation to avoid unnecessary complexity. That of course limits the capabilities of the demo but try to imagine what is possible with this technology! The goal is to make it easier for anyone with a Python background to try this out for themselves.

Let's start from the absolute basics: `index.html`.

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Document</title>
</head>
<body>
	<p>Hello, world!</p>
</body>
</html>
```

You can open `index.html` in a browser and you should see "Hello, world!".

What if I told you that you could add less than 20 lines of code to that `index.html` and have a working LLM to generate text? Here’s what that might look like:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<title>Document</title>
	<script type="module">
		import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1/dist/transformers.min.js';
		const generator = await pipeline('text-generation', 'Xenova/distilgpt2');
		const inputText = document.getElementById('inputText');
		const generateButton = document.getElementById('generateButton');
		generateButton.addEventListener('click', async () => {
		const text = inputText.value;
		const output = await generator(text);
		const outputDiv = document.getElementById('output');
		outputDiv.innerHTML = `<p>${output[0].generated_text}</p>`;
		});
	</script>
</head>
<body>
	<h1>Text Generation Example</h1>
	<p>Check the generated text below:</p>
	<input type="text" id="inputText" placeholder="Type your text here..." />
	<button id="generateButton">Generate Text</button>
	<div id="output"></div>
</body>
</html>
```

If that looks a little daunting then don't worry. Let's break down what's happening here, especially if you use Python and the Hugging Face Transformers library.

Keep in mind that it is "just" a basic text completion model (quite limited by today's standards) and is not trained to follow instructions as you would expect from ChatGPT etc. But let's try to look past the model's capabilities. Think about all the opportunities that come with making AI/ML models this accessible.
## If You Already Use Hugging Face in Python
If you’ve worked with Hugging Face Transformers in Python, the API should look familiar. You use a `pipeline` for common tasks like text generation, text classification, or question answering. In Python, it might look like this:
```python
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')
generator("Hello, world!")
```
In JavaScript, using Transformers.js, it’s almost the same idea:
```javascript
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1/dist/transformers.min.js';
const generator = await pipeline('text-generation', 'Xenova/distilgpt2');
```
The big difference is that this runs right in your browser. Everything is client-side. No backend server or API key is needed. Your data never leaves your device.
## Step by Step Breakdown
### 1. Importing the Pipeline
```javascript
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1/dist/transformers.min.js';
```
This line loads the Transformers.js pipeline from a CDN. No installation or setup. Just include it in your page.
### 2. Loading the Model
```javascript
const generator = await pipeline('text-generation', 'Xenova/distilgpt2');
``` 
Here, `'text-generation'` tells Transformers.js what task you want to perform, and `'Xenova/distilgpt2'` is the model you want to use. If you’ve loaded a model in Python, this should make sense right away.
### 3. Connecting to the Page
```javascript
const inputText = document.getElementById('inputText');
const generateButton = document.getElementById('generateButton');
```
This just grabs the input box and button from the HTML.
### 4. Adding the Click Handler
```javascript
generateButton.addEventListener('click', async () => {
	const text = inputText.value;
	const output = await generator(text);
	const outputDiv = document.getElementById('output');
	outputDiv.innerHTML = `<p>${output[0].generated_text}</p>`;
});
```

This part makes the app interactive. When the button is clicked, it reads the input text, runs the model, and shows the result.
## What’s Happening Under the Hood?

The first time you load the page, your browser downloads the model weights and saves them. This might take a few seconds the first time, depending on your connection. Once the model is ready, all the processing happens locally in your browser. No requests go to any server while generating text, so your data stays private.
## Why Use Transformers.js as a Python Developer?
Let's try to zoom out a little bit: Why do we train machine learning models? To be useful in solving a problem. If we look past the specific technologies or programming languages we are simply interested in integrating the models into systems in order for them to actually be used and hopefully deliver value to a user experience/process/etc.

I personally think there are a lot of use cases that could benefit from abandoning the "Python as the only tool" mentality that we see in machine learning integration and adoption. We want to use the right tool for the job, right? If that tool is Transformers.js and Javascript because it clearly makes the most sense to have client-side inference then let's choose that tool instead of shoehorning Python into the mix.

If you’re used to the Hugging Face Python library, Transformers.js can be an easy way to add AI features to web apps, quick demos, or share ideas with others. Some reasons you might want to try it:
- **Privacy**: Everything stays on the user's device. No data is sent to external servers during inference, eliminating concerns about sensitive information being transmitted or stored elsewhere.
- **Simplicity**: Deploy a working AI application with just plain HTML and JavaScript - no Python environment, package management or backend infrastructure needed.
- **Scalability**: The computation happens on users' devices, distributing the workload across your user base rather than centralizing it on your servers.
- **Free deployment**: Host your AI-powered web app anywhere that serves static files such as [Github Pages](https://pages.github.com/).
## What Can You Do Next?
This is just a very basic starting point. It gets a little more complicated once you want more functionality as you have to use [Web Workers](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers) but would recommend just sticking to the [tutorials](https://huggingface.co/docs/transformers.js/) in the documentation and reading those thoroughly. I can also just in general highly recommend checking out the [Transformers.js documentation](https://huggingface.co/docs/transformers.js/index), especially all of the [demos](https://huggingface.github.io/transformers.js/) and [examples](https://github.com/huggingface/transformers.js-examples) if you want to learn more.

If you would like to follow the project I can also recommend following its core developer Xenova on [Github](https://github.com/xenova/) or [HuggingFace](https://huggingface.co/Xenova/). He's often incredibly fast to implement demos of new features, models, etc. I really liked his [talk at WebAI Summit 2024](https://www.youtube.com/watch?v=n18Lrbo8VU8) which gives an excellent overview over the Transformers.js project. This talk on the [State of client side machine learning](https://www.youtube.com/watch?v=tF70o1Q8VkM) is also excellent.

Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/rasgaard/) and follow me on [Bluesky](https://bsky.app/profile/rasgaard.com) and let's have a chat about what you're building with Transformers.js or any ideas you might have!