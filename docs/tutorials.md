---
layout: listing
title: Tutorials
eyebrow: Work with Sensori
intro: Focused walkthroughs for turning raw wrist accelerometry into representations for downstream health research.
permalink: /tutorials/
---
<div class="tutorial-list">
  {% assign guides = site.tutorials | sort: 'order' %}
  {% for guide in guides %}
    <a href="{{ guide.url | relative_url }}" class="tutorial-row">
      <span class="tutorial-index">{{ guide.order | prepend: '0' | slice: -2, 2 }}</span>
      <span><strong>{{ guide.title }}</strong><small>{{ guide.subtitle }}</small></span>
      <span class="tutorial-detail">{{ guide.level }} · {{ guide.duration }}</span>
      <i aria-hidden="true">→</i>
    </a>
  {% endfor %}
</div>
