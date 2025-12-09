---
title: Data Scientist Technical Interview Case Walkthrough
date: 2024-02-13
tags:
  - post
---
# {{title}} 

During the interview process for a new job as a data scientist at Laerdal Copenhagen I was asked to complete a machine learning case study for the second round. The purpose was to engage in a technical discussion about my proposed solution and my associated considerations when completing the exercises. The case study itself was about getting an overview of sentiment and topics of image captions which would culminate in an analysis as well as a chat interface, utilizing a Large Language Model to interact with the data.

The case was divided into two exercises:

* Exercise A: Sentiments and Topics
* Exercise B: Combining Language and Images

I was sent a [GitHub repository](https://github.com/maximillian91/img-txt-categorisation-chat/) which provided instructions and technical details for downloading the data and some example code to load it. There were two datasets:

1) [COCO](https://cocodataset.org/#home) which contains images and a few associated captions.
2) [SentiCap](https://arxiv.org/abs/1510.01431) which contains alternative captions for the same images where each caption has been annotated with a sentiment score.

The code that I wrote for to solve the case can be found on my GitHub [here](https://github.com/rasgaard/laerdal-nlp-case-solution).

### Exercise A: Sentiment analysis

> Sentiment: Show in practice how you would apply a machine learning model on the captions to extract the sentiment (negative or positive) of them and future unlabeled captions.

The first task was to show how I would perform sentiment classification on the captions in order to get an overview of the sentiment distribution. My initial approach was to browse HuggingFace for an off-the-shelf model that could be used for sentiment classification. This had the advantage of being quick and easy to get an initial result but the disadvantage of not being trained on the specific distribution of data that it would be used on. Due to this disadvantage (and partly because I wanted to display that I could easily fine-tune a simple model) I decided to go against this approach.

Instead, I proposed to fine-tune a sentence-transformer using SetFit using only 32 samples per class from the SentiCap dataset. The classification accuracy of this model on the remainding captions (there are ~9,000 in total of which I used 64 to train the model) was about 95%. This allowed me to use a model that was fine-tuned on the specific domain and closer to the distribution of the COCO captions. I also uploaded it to the [HuggingFace model hub](https://huggingface.co/rasgaard/setfit-senticap) so that it can be easily used by others. The model card contains a lot of information out of the box when using SetFit!

  

<iframe src="../embeds/umap_embeddings.html" width="100%" height="650px"></iframe>

  

It should be noted that this approach might not be the best as forces each caption to be classified as either positive or negative. A lot of the captions in the COCO dataset are rather neutral and the very clear decision boundary in the UMAP plot might not be representative of the actual distribution of the captions. I suspect that the contrastive learning approach used by SetFit might contribute to the clear decision boundary because it forces the model to pull together captions of the same class and push apart captions of different class.

### Exercise A: Topic modelling

> Topic: Show in practice how you would apply a ML model on the captions to extract the topic(s) that they belong to.

Unlike the first task I wanted to do this one using off-the-shelf models and tools. Since I had briefly worked with [BERTopic](https://maartengr.github.io/BERTopic/index.html), I just wanted to show how easily it could be used to get started with topic modeling.. Simply following their first couple of Quick Start steps and various built-in visualization functions, you are able to extract topics from the captions and visually inspect them.

<div style="width: 1250px; margin-left:-200px;">

<iframe src="../embeds/topic_model.html" width="1250px" height="800px"></iframe>

</div>

  

I limited the number of topics to 200 as the visualization and overall usefullness would decrease in value if there were too many topics. You can explore each topic alone by double clicking on the topic on the right hand side of the plot. This will isolate the caption representations that are identified to be in the selected topic.

  

### Exercise B: Extracting captions from images

> Repeat step 1+2 with a ML model extracting information from the images valuable for determining the topic and sentiment (a pretrained Vision Language Model, VLM, could be beneficial).

  

> Compare the results from having captions only and then adding the image understanding.

  

At this stage, I encountered exercises that ventured into uncharted territory. I had never worked with Vision Language Models before and I was not sure how to approach this task. My first instinct was to go to HuggingFace to see if there was a category of models that could be used. To my relief I found that "Image to text" was a category and I found a model called "blip" developed by Salesforce. Using the DTU HPC GPU cluster I was able to run inference on all the ~40,000 images in the COCO dataset in order to generate captions.

  

Now, if I had followed the given instructions for the exercise, I would have repeated the two previous tasks with these newly generated captions. However, I had a hunch that the captions generated by the model would not necessarily contribute new information to the sentiment and topic modelling tasks. I decided to test this hunch by embedding the captions generated by the blip-model as well as the original COCO captions with a sentence-transformer model. These embeddings would then contain a numerical representation of the semantic information in the captions and could be compared with cosine similarity.

  

<iframe src="../embeds/similarity_scores.html" width="100%" height="650px"></iframe>

  

Since the cosine similarity between the original COCO captions and the blip-model captions was so high I decided to go against the exercise instructions and move on to the next exercise. This allowed me to spend more time on the final exercise which I found to be the most interesting and intriguing as I had been wanting a reason to work with local LLMs for a little while.

  

Another noteworthy observation about the blip-model captions was that they were homogeneous and not very descriptive. For example, it generated the caption "a train on the tracks" for 654 times. The same behaviour was seen with "a man playing tennis" and "a plate of food".

  

### Exercise B: Chatting about the images

> With an LLM in Langchain and Streamlit as the UI, write a Chat User Interface, where the user can have a dialog about their images, where the sentiment and topic is helping them and the chatbot to understand the images.

  

I had been looking forward to getting to this exercise as I had been wanting to work with local LLMs for a little while. As I wanted to be able to run everything locally (on an M2 16GB RAM MacBook Pro) I was fairly limited. I tried out [Ollama](https://ollama.com/) which was incredibly easy to get started with, just `ollama serve` and `ollama run llama2:7b-chat` and I was good to go.

As for the interface itself I had previously worked briefly with Streamlit but I knew that it should be easy as I had seen lots of examples of people creating these simple chat interfaces. I found [a repository](https://github.com/AustonianAI/chatbot-starter/) which provided a simple starting point which I could expand on. It used [LangChain](https://python.langchain.com/docs/get_started/introduction) as well to communicate with the LLM which was perfect as I wanted to try that out as well.

Now that I had a working chat interface that could communicate with a local LLM I had to remind myself of what the original task was. The task was to create a chat interface where the user could have a dialog about their images and the sentiment and topic of the captions would somehow help the chatbot understand the images. I divided the task into two subtasks:

1. Detecting when the user would like to pull up an image and display it

2. Integrating the caption along with sentiment and topic into the model's context and using it to generate a response

  

#### Subtask 1: Detecting user's intent

For the first subtask I thought that it would be fun to use the LLM itself to detect whether the intention of the user is to find an image. The resulting function is practically just a prompt that detects whether the user is asking to see an image by providing a few examples of how the user might ask for images. The prompt is as follows:

  
```plain
Answer ONLY the parsable token that is specified in the prompt.

If the user DOES NOT request an image you should respond with a message that says "<NOREQ>".

Here are some examples of NOT requesting an image:
- "I'd like to know what a cat looks like."
- "Can you describe a bicycle to me?"
- "I want to know what a mountain looks like."
- "Where is the Zebra located?"
- "Can you tell me anything else about New York?"

  

If the user DOES request to see an image you should answer "<IMGREQ>".

Here are some examples of image requests:
- "Can you show me a cat?"
- "Find me an image of a bicycle."
- "I'd like to see a picture of a mountain."
- "I want to see a photo of a dog."
- "Find a picture of a woman riding a bike."
- "Hello, can you find me a photo of a man?"
```
  

I had seen examples of using LLMs for this kind of task before but had never actually tried it out myself. It was pretty fun to see for myself that it actually worked quite well.

#### Subtask 2: Locating and using appropriate context

The motivation for this subtask was to use all the appropriate information about the image that the user is requesting to chat about in order to generate a meaningful response. What I'm alluding to here is that we need some sort of Retrieval Augmented Generation (RAG) system in order to retrieve a relevant image and its caption. I had heard about vector databases but thought I would try a simpler approach of just using a numpy array which was inspired by this [tweet from Andrej Karpathy](https://twitter.com/karpathy/status/1647374645316968449).

  

The idea was to find the most similar captions in the COCO dataset based on the user's request and display the corresponding image. I used an off-the-shelf sentence-transformer model to embed the captions and store them in a numpy array. I then used the same model to embed the user's request and found the most similar captions using cosine similarity. The corresponding image were then displayed to the user and the caption would be used to formulate a response from the chatbot. The prompt for the chatbot was as follows:

  
```plain
Your job is to describe the image in a way that is useful to the user.

The image will be described to you and you will need to describe it back to the user.

Always start your response with "Here's a picture of " and then add a description of the image.

If the user query and image description DOES NOT match you should provide an explanation as to why the user's query might match the image description.
```
  
  

![Example of image retrieval in the chat interface](attachments/chat_example.png)

  

Amazingly, the chatbot was able to correctly identify if the user was asking to see an image, retrieve a relevant image and describe it in detail from the image's caption as well as the overall sentiment. I forgot to include information about the topic as I was just so excited that it actually worked out in the end but could easily extend the chatbot to include this as well.

  

### Conclusion

  

Initially, I was nervous about the last few exercises as they posed new challenges but I was able to overcome them and produce a working solution which I am proud of. Overall, the case was a great opportunity to work with and showcase the skills I have been developing over the past few years when I decided that I would like to focus on NLP and machine learning.