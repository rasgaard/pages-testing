# Blog posts

{% for post in collections.posts %}
- **{{ post.date | niceDate }}** — <a href="{{ post.url }}">{{ post.data.title }}</a>
{%- endfor %}
