---
title: Transferring health information
subtitle: Linear probes trained in one cohort retain predictive signal across populations, countries and measurement settings.
summary: Frozen representations generalise across UKB, CKB, ELSA and NHANES, including harmonised health traits and physical function.
kicker: Cross-cohort transfer
number: "03"
order: 3
accent: blue
signal: [16, 28, 24, 50, 64, 59, 76, 68, 42, 71, 52, 30]
hero_label: Population cohorts
hero_stat: "4"
hero_caption: UKB, CKB, ELSA and NHANES
tags: [Generalisability, Population health, Physical function]
back_url: /research/
back_label: All research
next_link: /research/disease-prediction/
next_label: Next case study
next_title: Diagnosing and predicting disease
---

## Generalisation is the test

Large datasets make it possible to learn expressive models, but scale alone does not establish that a representation will travel. Wrist devices, study protocols and participant characteristics differ across cohorts. A useful population representation must retain meaning across those changes.

## A deliberately strict setup

For harmonised targets, linear probes were fitted in the UK Biobank training set and applied without refitting in held-out UKB, CKB, ELSA and NHANES data. This separates the quality of the representation from the flexibility of downstream adaptation.

Sensori representations predicted age, sex and body mass index across the four cohorts. They also captured smoking, alcohol consumption and self-rated health across the cohorts in which those measures could be harmonised.

## From health state to function

The signal extended to practical measures of daily function. In ELSA and NHANES, representations were informative for mobility and other activities of daily living, including preparing meals and attending social events.

These results do not make the cohorts interchangeable. They show that a common layer of health-related movement information is preserved across populations and measurement settings—an essential property for models intended to support research beyond their development dataset.
