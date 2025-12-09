# Status Updates

{%- for status in collections.status %}
- {{ status.data.year }}, **{{ status.data.month }}** — [{{ status.data.title }}]({{ status.url}})
{%- endfor %}