---
layout: listing
title: Research
eyebrow: The evidence
intro: From minute-level activity to disease prediction, these studies test what a representation of an entire day can reveal.
permalink: /research/
published: false
---
<div class="project-list listing-projects">
  {% assign items = site.research | sort: 'order' %}
  {% for item in items %}{% include research-card.html item=item %}{% endfor %}
</div>
