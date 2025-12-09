---
title: Enhancing Customer Service with Fine-Tuned Sentence-Transformers
date: 2023-07-17
tags:
  - post
---
# {{title}}
## Background
I was employed at Danske Bank for about two years (Oct. 2021 to Aug. 2023) as a student Machine Learning Engineer during my MSc studies. I worked a lot with on-prem infrastructure planning and deployment as well as tool assessments and integrations. Sometimes I also did some data science work (i.e. try to obtain insights from data) but mostly I supported the data scientists such that they could focus on their time and not have to think about infrastructure, scheduling, etc. MLOps would be a suitable word for my daily tasks.

The on-prem tools that were available did not support data science work very well as resources were scarce and infrastructure was scattered. This of course posed lots of challenges when doing MLOps. I spent a lot of time purely on assessment of the capabilities of the offered infrastructure and what the limits were in terms of computation in order to set expectations for what kind of models could be deployed.
## It all began with a "Hackathon"

During the summer of 2022 our team had been approached by a Finnish customer service team who presented a problem that they thought we might be able to help them solve. Due to them being short on staff they had worked up a huge backlog of unanswered messages. This of course hurts the customer service experience immensely and should be avoided but had happened due to a mismatch in staff count and volume of incoming messages. They proposed to offer us access to data and then we could work on a proof of concept.

Since many messages were low-stakes questions such as "How do I open new account?", "How do I order a new card?", "How do I get a card for my child?" (as opposed to high-stakes questions related to fraud, identity theft, etc.) those could just as well be answered by pre-written self-service instructions. This way we could help the subject matter expert's by freeing up their time, allowing them to answer more pressing issues and provide a good customer experience for those cases.

The development of the proof of concept was presented as a Hackathon by management. I think it made it more lighthearted and fun to think about and also encouraged working together on a single project. Seeing as this was during the summer and lots of colleagues were on vacation we had the ability to pause other projects to focus intensely on the Hackathon for a short while.
## We need labels..?

Starting off we had a lot of different ideas of what the best plan of action would be to take. Some took an unsupervised approach with topic modelling while others went in a supervised direction which required labels for the text. Ultimately, we opted for the supervised approach as the data was perhaps a bit too noisy for an unsupervised approach to yield meaningful results without lots of intervening.

One thing I wish we would have done differently is to slow down a bit and start with *simpler* solutions but seeing as we were all in Hackathon-mode we wanted things to move quickly using the coolest tools available. This of course also served as an excuse for many that just wanted to familiarize themselves with e.g. Transformer-based models such as RoBERTa.

### Bulk labelling

The first approach to labelling came in the form of bulk labelling - taking heavy inspiration from [bulk](https://github.com/koaning/bulk) by Vincent Warmerdam. This had the great advantage of being incredibly intuitive, allowing subject matter experts to label data themselves and using their own intuition and domain knowledge to provide labels. We used these labelled messages to fine-tune a classifier with RoBERTa and got great results!

## Pivoting to new methods

This served as a good learning experience and essentially showed that classifying the messages into categories was possible but it was far from the actual product that we would eventually produce. Several things were problematic with this approach:

1. Labels did not necessarily match intended functionality
	- For example, using the bulk labelling tool we might identify and label a category of messages that we don't intend on ever answering automatically, making labels for those messages meaningless to some extent.
2. Long training time on development platform
	- We were working on remote virtual machines and had no access GPU resources at the time.
3. Insufficient compute on deployment platform
	- The Kubernetes cluster we used for deployment provided a very small amount of computational resources for each pod.

At this point in time I felt we needed to somehow pivot in order to fulfill the wishes of the customer service team. Also during this time, I had recently stumbled upon [SetFit](https://huggingface.co/blog/setfit) which showed great potential by fine-tuning sentence-transformers with very few labelled data points. I thought that this is exactly what we needed but initially had a hard time convincing my colleagues to try out this new method.

  

Eventually I took it upon myself to track down the data and attempt to replicate the results from the RoBERTa model but only with a small fraction of the data. This managed to persuade my colleagues into thinking that this new method was actually worth investing time and effort into as it provided a way to solve the aforementioned problems by

  

1. Making the labelling process simpler through requiring much fewer labels per class

2. Having less data to train on also meant that training was faster. The [model](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) itself was also considerably smaller.

3. Using [Optimum](https://github.com/huggingface/optimum) the fine-tuned model was able to be optimized and actually run in near real-time on the deployment platform.

  

The process was now that 1) the Finnish customer service team identified which topics they would like for the classification system to be able to categorize, 2) then we would offer a [labelling platform](https://github.com/doccano/doccano) on which subject matter experts would visit and label messages according to the identified subjects and 3) we would train a model and [expose an API](https://fastapi.tiangolo.com/) where a message ID could be passed and a category would be returned together with its softmax probability of being in that class. The message routing system would then use the category together with probability to assess whether the system was certain that the message belonged to the category and send out an automated response if there was a strong sign of confidense.
## Closing off

This project arrived at a fantastic time as I wouldn't have been familiar with the necessary tools and infrastructure just a few months prior. Not only that but SetFit was also in early development. If it weren't for the quick iteration speed that SetFit allowed we would have been limited to progress at a much slower pace.

However, a slower pace - especially in the early phases - might have been beneficial as we would perhaps make more careful considerations and try out simpler methods. As time progressed I was made increasingly aware that methods from information retrieval such as BM25 might perform just as well as many of the more complex deep learning based approaches.

In conclusion, it was a fantastic learning experience as it involved lots of stakeholders that were interested and helpful in developing a product that utilized the team's machine learning capabilities. My personal biggest takeaway is perhaps that taking machine learning projects out of the notebooks and into systems where they can actually have a real-world impact is challenging to a degree I did not expect but ultimately incredibly rewarding.