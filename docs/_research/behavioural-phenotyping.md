---
title: Recovering everyday behaviour
subtitle: A frozen day-level embedding captures activity, walking, sleep and device characteristics across population studies.
summary: Simple probes recover 18 conventional wearable-derived traits while retaining information beyond those predefined summaries.
kicker: Daily behaviour
number: "02"
order: 2
accent: violet
signal: [22, 20, 18, 35, 48, 72, 66, 53, 84, 70, 58, 44]
hero_label: Behavioural targets
hero_stat: "18"
hero_caption: Activity, walking, sleep and device-derived traits
tags: [Behaviour, Sleep, Linear probing]
back_url: /research/
back_label: All research
next_link: /research/cross-cohort-health/
next_label: Next case study
next_title: Transferring health information
---

## Beyond a checklist of traits

Wearable research often begins by reducing a recording to a small collection of familiar quantities: steps, activity intensity and sleep duration. These measures are useful and interpretable, but each is selected in advance and compresses away most of the original signal.

We asked a complementary question: if Sensori learns a general representation from the raw day, can conventional behavioural traits be recovered from it afterwards?

## Eighteen views of a day

We evaluated six activity measures, five walking measures and seven sleep measures. Linear probes were fitted using day-level Sensori embeddings from the UK Biobank training set and applied without refitting to held-out and external cohorts wherever the same target was available.

The embeddings predicted device-measured behavioural traits more accurately than demographic covariates alone across the cohorts. They also outperformed basic device statistics, indicating that their information cannot be reduced to mean movement intensity and variability.

## A representation, not a replacement

Interpretable behavioural measures remain important for communication and clinical reasoning. Sensori offers a different layer: a reusable representation that can be queried for a target after pretraining, without deciding beforehand which parts of the day matter.

This distinction is central to the project. The aim is not to replace steps or sleep duration, but to make the rest of the signal available for analysis.
