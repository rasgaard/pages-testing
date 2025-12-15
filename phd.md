---
layout: default
title: PhD Status
---

# Status Updates

{% for status in site.phd_status %}
   - {{status.year}}, **{{status.month}}** - <a href="{{status.url}}">{{status.title}}</a>
{% endfor %}