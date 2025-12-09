# Favorites

{% for item in collections.favorites %}
- Week **{{ item.data.week }}** — <a href="{{ item.url }}">{{ item.data.title }}</a>
{%- endfor %}
