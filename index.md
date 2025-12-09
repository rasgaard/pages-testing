---
layout: default
title: My Blog
---

# Welcome to My Blog

This is a simple GitHub Pages site experimenting with Jekyll.

## Recent Posts

<ul class="post-list">
{% for post in site.posts %}
  <li>
    <div class="post-title">
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    </div>
    <div class="post-date">{{ post.date | date: "%B %d, %Y" }}</div>
  </li>
{% endfor %}
</ul>
