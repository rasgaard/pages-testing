---
title: EurIPS, Course development and holidays
layout: default
date: 2025-12-31
tags:
  - status
---
# {{ page.title }}

## EurIPS
First week of december was spent attending EurIPS. I had been looking forward to this conference quite a lot. I believe it is the largest machine learning conference I have attended thus far. Getting a taste for the scale and reach of the machine learning community was fantastic. I took some notes at the end of each day based on my experience.
### Day 1: ellis UnConference
I participated in the workshop titled *"Energy-Efficient AI: Models, Algorithms, and Hardware for Sustainable Intelligence"*. 

My main takeaway from the talks presented at the workshop was that *neuromorphic* computing is an interesting approach to dealing with limitations of hardware. 

My favorite talk was "*Analog In-Memory Computing for Efficient Large Language Model Deployment*" by Iason Chalas (ETH and IBM Zurich) because it took this idea that I have vague familiarity with from an old [Veritasium video](https://www.youtube.com/watch?v=GVsUOuSjvcg) and put it into context of modern LLMs. The main idea, as I understand it is that current can flow in parallel through the analog system, making it more efficient than serial digital computations.

I also really liked "*Two steps forward and no steps back: Training neural networks in noisy hardware without backward passes*" by Nasir Ahmad (Radboud University Nijmegen) who presented some weird yet intriguing ideas. He built up from the fact that inference is *much* cheaper than backprop. Almost 100x at this point for some accelerators (Cerebras, Groq). What if we just take random steps and evaluate models that way? Will that be as (or more) efficient that traditional backprop? I'm wondering whether LoRAs can play a role in efficiently "sampling" new models.  

I didn't pay much attention during the panel discussion. One guy asked whether it even makes sense to approach AI *neuromorpically*. That resonated with me quite a bit. You can do all kinds of clever tricks informed by how (we think!) the brain works but it will only get you so far.

I also randomly bumped into a guy whose LinkedIn tagline is "LLM Neurosurgeon". He works with quantization, particularly quantization aware training (QAT).

The poster session was *fine*. I trembled a bit while briefly presenting at Lenka's paper. Otherwise I just walked around aimlessly for a bit. The best part was definitely just talking to people from the section.

I got a coffee cup on the way out. That's a big win in my book.

### Day 2: First officielt EurIPS day
Now we're moving on to the actual show. Yesterday was planned by ellis - today is EurIPS. All presented work from now on is accepted at NeurIPS.

The day started off with a presentation round in the main hall. My favorite was *"Differentiable Sparsity via $D$-Gating: Simple, Modular and Versatile Structured Penalization"* by David Rügamer. Although I didn't completely get the method - even after talking to the presenter at the poster - I get the sense that it's a very promising method to induce structured(!) sparsity into a model.

The first ever EurIPS keynote was "*Adaptive Bayesian Intelligence and the Road to Sustainable AI*" by Emtiyaz Khan and being someone who's definitely not in the Bayesian mindset, I could appreciate all of his illustrations and animations that visualized how the ideas and methods worked.

I also got to chat with some people that I've otherwise only interacted with online. That was pretty nice.

After the keynote there was a panel discussion about being a researcher/entrepreneur. How it's currently the best time to be in both camps in terms of opportunities. I was pretty tired at this point and didn't get much from this discussion. 

We stayed until 6pm to catch Richard Suttons keynote which was streamed from San Diego. He presented a framework that does not fall short in terms of his bitter lesson. He kept mentioning "finding the rest room" as a task for an agent acting in an environment so we thought that he might have needed to go during the talk.

Takeaway from today: Feeling some imposter syndrome. A lot of work goes into all the research being presented. Starting to wonder if I'll be able to put in an equivalent amount of work towards something as excellent as these people.

### Day 3: Posters and LLM jailbreaking
This morning I bookmarked 13 posters to check out. It was unrealistic to get through all of that but I managed to talk to a few authors.

During lunch I randomly bumped into Nicki who told me that the conference really managed to get some heavy hitters despite the short time schedule. He was referring to the next keynote speaker, Sepp Hochreiter, who came up with the LSTM. 
The keynote was about how scaling transformers is not a sustainable path forward. His group/company recently proposed xLSTM - a way to effectively scale LSTMs to billions of parameters. 

Much of the afternoon was spent on a game built by Lakera, one of the company sponsors. They specialize in LLM security such as prompt injection prevention. Their game was a simulated environment where you had to inject prompts into a paper where the premise is that the conference is using an LLM-based auto-reviewer system. You would have to manipulate this system in order to score points. The prize for getting for most points was a Nintendo Switch 2.

Viktor shot up the leaderboard quite quickly and figured out some clever ways to get through the harder levels. His main competition on the top of the leaderboard was Santiago, a Principal Researcher at Microsoft who would be giving a talk the following Saturday at a workshop on LLM security.

He managed to get the final points during the conference dinner, putting him in a shared 1st place with Santiago with the maximum amount of points.

### Day 4: Competition Drama, Standing Up to Scale and EurIPA
For the last day I decided to skip the paper talks during the morning session and joined the keynote by Judith Rousseau. I didn't really pay attention to the keynote titled "*Bayesian semiparametric inference: a blessing or a curse.*". As one person during lunch put it she "dumped her appendix on some slides and called it a keynote". A lot of equations that were hard to follow.

Salon des Refusés, a poster session with declined papers, was after the keynote. My favorite one was a position paper that discussed that technical explanations that XAI researchers produce are rarely, if ever, useful for people that don't develop models. I sort of realized this while working with XAI during the later parts of my master's degree. I think that XAI can benefit from a shift in perspective into being viewed as a diagnosis tool for model developers.
 
 At 12:00 the Nintendo Switch 2 competition would end and we were standing at their booth to see what would happen. Santiago, the Principal Researcher from Microsoft also showed up. We had anticipated that we would probably pull out of the competition and let Viktor have the prize but that was not the case. It was a draw and the competition organizers had built an application to do a very inefficient coin flip. Viktor won by a narrow margin.

The final keynote was by Gaël Varoquaux, the creator of scikit-learn. It was a fun, yet serious, presentation about how scientists should push back on unrealistic reviews that demand hyperscaler-like resources to satisfy.

The final remarks by Søren Hauberg was fantastic. He was genuinely grateful for all of the support and the can-do attitude of everyone involved. One of his favorite moments was when the lights randomly went off during a poster session and people quickly pulled up their phones, turned on the flashlights, and continued the discussion. He also announced that during these past few months he have had to send a lot of late night texts, sometimes battling autocorrect and fat-fingering the smartphone keyboard. That had let to the fortunate mistake of typing "EurIPA" which led to the idea that there should be a special brew for the occasion.



## Course development
In January I will be joining the teaching assistant crew in my supervisors (Nicki) wildly popular [MLOps course](https://skaftenicki.github.io/dtu_mlops/). At the time of writing the course has a whopping 450 registered attendees. It'll be super interesting to talk to the students to gauge why they took this course specifically.

Scaling a course is an interesting and important challenge. A lot of the courses at DTU Compute are facing challenges with scale as they are fundamental for further studies in specialized fields. That means we have students from practically all fields of the university taking courses at this department. 

The course is set up to scale quite well to begin with. It's simply a collection of hands-on exercises that are easily accessible to everyone on the course website. The course then structures the work that goes into doing the exercises and facilitates a project. 

The final piece of the scaling-puzzle relates to how students can get reliable and accurate feedback on an on-going basis. Nicki had experimented with an approach that would scrape the relevant repositories and evaluate them based on criteria such as "Are there Github Actions workflow files for CI/CD?", "Are requirements specified in requirements.txt or pyproject.toml?", "How many lines of code?", etc. My job has been to collect those scripts build up a package that provides a cohesive environment for further development. That has resulted in [https://github.com/rasgaard/mlops-mentor/.](https://github.com/rasgaard/mlops-mentor/.) 

It is not completely done and I'm not totally comfortable using it as an official tool just yet. I would like to co-create it with the students during the course if possible. But, there are signs that it could be highly useful for the project phase. 

## Holidays
Last, but definitely not least, I spent time during the holidays to wind down and reflect on the year that has passed. I'm proud to say that I didn't read a single paper during these (almost) two weeks of vacation. It has been nice but I'm also ready to get back into the groove. 
