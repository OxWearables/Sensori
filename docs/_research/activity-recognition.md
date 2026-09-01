---
title: Recognising movement in the wild
subtitle: Minute-level representations transfer to four external activity-recognition benchmarks without task-specific representation learning.
summary: Sensori's minute-level embeddings retain local movement structure across both controlled and free-living settings.
kicker: Local movement
number: "01"
order: 1
accent: coral
signal: [18, 34, 62, 88, 51, 30, 74, 46, 92, 58, 25, 42]
hero_label: External benchmarks
hero_stat: "4"
hero_caption: PAMAP2, RealWorld, WISDM and CAPTURE-24
tags: [Representation learning, Activity recognition, Transfer]
back_url: /research/
back_label: All research
next_link: /research/behavioural-phenotyping/
next_label: Next case study
next_title: Recovering everyday behaviour
---

## The question

A whole-day representation is only useful if it preserves the smaller events from which a day is composed. We therefore tested whether the model's minute-level embeddings retained enough local information to separate human activities.

## The evaluation

Sensori was evaluated on PAMAP2, RealWorld, WISDM and CAPTURE-24. These datasets span structured activity protocols and unconstrained free-living recordings. A regularised linear classifier was fitted to frozen minute-level embeddings using participant-wise cross-validation, ensuring that recordings from the same person never appeared in both training and evaluation folds.

> The probe is deliberately simple: strong performance should reflect information already present in the representation, not a powerful downstream classifier.

## What we found

Sensori achieved the highest mean Cohen's κ on PAMAP2 and RealWorld, and the second-highest on WISDM and CAPTURE-24. The result suggests that pretraining on continuous population recordings produces features that remain useful at the scale of individual activities.

## Why it matters

The same model can support analyses at two distinct resolutions: minute-level representations for local activity and day-level representations for health. This hierarchy avoids defining the value of a movement signal through a single handcrafted summary.
