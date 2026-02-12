---
layout: default
title: PhD Status
---
## Status Updates
Once a month I try to sit down and think about what has happened.

{% assign posts = site.phd_status | sort: 'date' | reverse %}
{% for status in posts %}
   - {{status.date | date: '%Y'}}, **{{status.date | date: '%B'}}** - <a href="{{status.url}}">{{status.title}}</a>
{% endfor %}
## About the project
It is titled *"Edge deployment of deep neural networks via Model Compression for Healthcare Applications"* and is a collaboration between [Laerdal Medical](https://laerdal.com/) and the [Technical University of Denmark](https://www.dtu.dk/). It's a 3-year PhD going from October 1st 2025 to September 31st 2028.

Here are a few key statements that motivates me about the project: 
- We'd like to see how much performance we can squeeze onto devices with limited compute.
- Huge amounts of resources have already one into training models. Let's make sure that they're put to use by making them applicable on a wide range of devices.
- Centralizing AI inference is the standard but it's not always clear winner. *Distributing* inference compute to the client-side can be hugely beneficial, perhaps even necessary, due to privacy, latency or cost at scale.