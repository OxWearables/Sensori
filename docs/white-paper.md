---
layout: white-paper
title: Movement is a language of health.
eyebrow: The Sensori white paper
author: Hang Yuan, Yong Wang
date: 2026-08-13
premise: Much of human health and function unfolds beyond the clinic, through the movements of everyday life.
description: The principles, lessons and broader vision behind the Sensori movement foundation model.
permalink: /white-paper/
---

## Where it all began
The term *foundation model* was only introduced in 2021 by [Bommasani and colleagues at Stanford](https://arxiv.org/abs/2108.07258). Although the ML community was already familiar with the benefits of large-scale pretraining, the idea was still new to wearable sensing. Large wearable datasets, particularly movement data, were already available through NHANES and UK Biobank, but, as with many new ideas, the field needed time to catch up. After publishing [one of the first large-scale pretrained models for movement](https://www.nature.com/articles/s41746-024-01062-3), using a simple multi-task objective to predict transformations of the input signal, we came across [Apple's foundation models for PPG and ECG](https://machinelearning.apple.com/research/large-scale-training). Trained on the Apple Heart and Movement Study, these models showed a striking ability to distinguish participant characteristics and a range of existing health conditions. At first, the results seemed almost too good to be true. That made us want to test whether the same idea could work with publicly available UK Biobank data and wrist movement as the input modality.


## Why movement matters
Passive movement monitoring is universal. Approximately one in three adults who use the internet owns a wrist-worn device capable of measuring movement. If we can recover even a small fraction of the health information carried by these signals, the benefits at population scale could be substantial. This ambition is deeply rooted in the philosophy of Oxford's Nuffield Department of Population Health, the birthplace of UK Biobank: to understand simple but important modifiable risk factors, such as smoking, so that targeted interventions can deliver meaningful benefits across entire populations.

For movement sensing, our intuition is simple:
> A direct way to assess someone's health is to follow them for a day and observe what they do, and how they do it.


## From behavioural traits to learned representations
Sensori takes a different approach. Instead of reducing wrist movement to predefined behavioural traits, such as step count or gait asymmetry, it treats the full 24-hour waveform as a continuous time series. We hypothesised that self-supervised learning could discover health-relevant patterns across timescales without first deciding which behaviours matter. This differs from Apple's [WBM](https://proceedings.mlr.press/v267/erturk25a.html) and Google's [SensorFM](https://arxiv.org/abs/2605.22759), which learn from derived behavioural or sensor features. Such summaries remain useful, but choosing in advance what to retain can limit the information available to the model.


The comparison below makes this distinction concrete. Even 450 conventional features—18 daily traits and 432 hourly summaries spanning physical activity, steps and sleep—were substantially less predictive of sex, age and BMI than Sensori embeddings learned directly from the raw signal. The same pattern extended across ten additional health axes tested in the UK Biobank.

<figure class="representation-evidence" aria-labelledby="representation-evidence-title">
  <header class="representation-evidence-header">
    <p>From summaries to representation</p>
    <h3 id="representation-evidence-title">Sensori captures more than predefined summaries.</h3>
  </header>
  <div class="representation-outcomes" aria-label="Performance gains for sex, age and body mass index">
    <section class="representation-outcome" aria-labelledby="representation-sex-title">
      <div class="representation-outcome-heading">
        <h4 id="representation-sex-title">Sex</h4>
        <span>AUROC gain</span>
      </div>
      <div class="representation-outcome-chart" role="img" aria-label="Sex AUROC gain: 0.185 for daily traits, 0.115 for hourly traits, 0.203 for combined daily and hourly traits, and 0.355 for the Sensori representation.">
        <div class="representation-outcome-bar">
          <strong>+0.185</strong><i style="--bar-value: 30.8%"></i><span>Daily<small>18 traits</small></span>
        </div>
        <div class="representation-outcome-bar">
          <strong>+0.115</strong><i style="--bar-value: 19.2%"></i><span>Hourly<small>18 × 24 (432)</small></span>
        </div>
        <div class="representation-outcome-bar">
          <strong>+0.203</strong><i style="--bar-value: 33.8%"></i><span>Both<small>450 total</small></span>
        </div>
        <div class="representation-outcome-bar representation-outcome-bar-sensori">
          <strong>+0.355</strong><i style="--bar-value: 59.2%"></i><span>Sensori</span>
        </div>
      </div>
    </section>
    <section class="representation-outcome" aria-labelledby="representation-age-title">
      <div class="representation-outcome-heading">
        <h4 id="representation-age-title">Age</h4>
        <span>Pearson <i>r</i> gain</span>
      </div>
      <div class="representation-outcome-chart" role="img" aria-label="Age Pearson correlation gain: 0.143 for daily traits, 0.285 for hourly traits, 0.293 for combined daily and hourly traits, and 0.542 for the Sensori representation.">
        <div class="representation-outcome-bar">
          <strong>+0.143</strong><i style="--bar-value: 23.8%"></i><span>Daily<small>18 traits</small></span>
        </div>
        <div class="representation-outcome-bar">
          <strong>+0.285</strong><i style="--bar-value: 47.6%"></i><span>Hourly<small>18 × 24 (432)</small></span>
        </div>
        <div class="representation-outcome-bar">
          <strong>+0.293</strong><i style="--bar-value: 48.8%"></i><span>Both<small>450 total</small></span>
        </div>
        <div class="representation-outcome-bar representation-outcome-bar-sensori">
          <strong>+0.542</strong><i style="--bar-value: 90.3%"></i><span>Sensori</span>
        </div>
      </div>
    </section>
    <section class="representation-outcome" aria-labelledby="representation-bmi-title">
      <div class="representation-outcome-heading">
        <h4 id="representation-bmi-title">BMI</h4>
        <span>Pearson <i>r</i> gain</span>
      </div>
      <div class="representation-outcome-chart" role="img" aria-label="Body mass index Pearson correlation gain: 0.105 for daily traits, 0.110 for hourly traits, 0.134 for combined daily and hourly traits, and 0.461 for the Sensori representation.">
        <div class="representation-outcome-bar">
          <strong>+0.105</strong><i style="--bar-value: 17.5%"></i><span>Daily<small>18 traits</small></span>
        </div>
        <div class="representation-outcome-bar">
          <strong>+0.110</strong><i style="--bar-value: 18.3%"></i><span>Hourly<small>18 × 24 (432)</small></span>
        </div>
        <div class="representation-outcome-bar">
          <strong>+0.134</strong><i style="--bar-value: 22.4%"></i><span>Both<small>450 total</small></span>
        </div>
        <div class="representation-outcome-bar representation-outcome-bar-sensori">
          <strong>+0.461</strong><i style="--bar-value: 76.8%"></i><span>Sensori</span>
        </div>
      </div>
    </section>
  </div>
  <div class="representation-breadth" aria-label="Across ten health axes, mean AUROC gain was 0.048 for Sensori compared with 0.026 for combined daily and hourly traits.">
    <span>Across 10 health axes</span>
    <p><strong>+0.048</strong> Sensori <i>vs</i> <strong>+0.026</strong> daily + hourly traits</p>
  </div>
  <figcaption>Gain over acceleration mean and s.d. in held-out UK Biobank; point estimates shown.</figcaption>
</figure>


## How we developed Sensori
Sensori was designed around a simple premise: movement carries health information across multiple temporal scales. A tremor may unfold over seconds, while the relationship between daytime activity and night-time sleep emerges across the full day. Capturing both requires a model that can learn local movement patterns alongside characteristics that remain stable over time.

We therefore combined two complementary pretraining objectives. [Masked reconstruction](https://aclanthology.org/N19-1423/) hides selected five-minute segments and asks the model to recover them from the surrounding movement, encouraging it to learn within-day temporal structure. [Participant-level contrastive learning](https://proceedings.mlr.press/v119/chen20j.html) brings recordings from different days of the same person closer together while separating those from different people, encouraging the representation to retain stable individual characteristics. Because each participant contributed at most seven complete days, this contrastive objective was prone to overfitting. We therefore introduced it only during the final stage of pretraining, after masked reconstruction had established a robust representation of within-day movement.


## What to use Sensori for

### Health monitoring
Medicine has traditionally focused on detecting and treating established disease. Preventive care also requires us to measure health before disease emerges. Tracking how health changes over time could reveal early deterioration, help evaluate interventions and extend the years people live in good health.

Movement is conceptually simple but information-rich. Sensori embeddings captured variation spanning participant characteristics, smoking status, chronotype, self-rated health and physical function. Many of these domains are ordinarily assessed through questionnaires or occasional clinical measurements. Movement representations could complement these assessments with low-cost, repeated measurements at population scale, enabling new research into health trajectories and supporting future health and fitness applications.


### Disease diagnosis and prognosis
Population-scale disease prediction is difficult because many conditions affect fewer than 1% of participants, creating severe class imbalance and leaving relatively few labelled examples. Sensori separates general representation learning from disease-specific learning: it first learns common movement patterns from a large unlabelled dataset, then adapts to each condition through supervised fine-tuning. In UK Biobank, adding fine-tuned Sensori embeddings to common clinical covariates improved prevalent disease classification for 52 of 102 eligible conditions and six-year incident disease risk prediction for 26 of 87. The first task asks whether current movement reflects existing disease; the second asks whether it helps anticipate future onset.

Sensori was deliberately designed to be general-purpose and disease-agnostic, but no established benchmark covered this breadth of conditions. We therefore compared its embeddings with progressively richer conventional movement inputs, from acceleration statistics to 18 device-measured behavioural traits. These comparisons do not establish a formal upper bound, but they help indicate how much predictive information is available in wrist movement. Conventional traits produced smaller gains for prognosis than for prevalent disease classification, suggesting that many future outcomes are less directly expressed in movement at a single point in time or that much of the cause-effect association from movement was absorbed by body mass index (BMI), which was used as a covariate.

Although we evaluated a broad range of diseases, the largest gains were concentrated among conditions with movement-related phenotypes, particularly neurological and psychiatric disorders. Movement alone will not capture every aspect of disease. Combining Sensori with complementary modalities, such as electrocardiography, electronic health records or proteomic measurements, could provide a more complete view of disease status and future risk.

### Digital biomarker discovery
Digital biomarkers are objective measures derived from digital devices that can reflect health status, disease progression or response to treatment. A familiar analogue is BMI: its trajectory can prompt preventive lifestyle changes, while persistently high values may help inform more intensive treatment. A useful digital biomarker could play a similar role by turning passively collected signals into a measure that can be followed over time.

Developing such measures can take years. Researchers must first identify promising candidates among many possible features, then establish that a selected measure tracks the relevant aspect of disease reliably across devices, environments and populations.

A more fundamental question comes first: does the signal contain enough information to distinguish the health states of interest? A carefully designed movement measure may still add little beyond a questionnaire or common clinical variables. Without first establishing the information available in the underlying signal, biomarker development can become a costly search with an uncertain ceiling.

Sensori can help guide this search. When its embeddings discriminate a disease, they provide evidence that the raw movement signal contains relevant information and identify conditions for which biomarker discovery may be most promising. The embedding model can then serve as a benchmark while researchers search for simpler, interpretable measures that recover the same signal. An [agentic AI framework](https://ui.adsabs.harvard.edu/abs/2026arXiv260414615K/abstract) could accelerate this process by testing large numbers of candidate measures, but any resulting biomarker would still require independent and prospective validation.

<figure class="biomarker-pathway" aria-labelledby="biomarker-pathway-title">
  <header class="biomarker-pathway-header">
    <p>Discovery pathway</p>
    <h3 id="biomarker-pathway-title">From signal to biomarker.</h3>
  </header>
  <ol class="biomarker-pathway-stages">
    <li class="biomarker-pathway-stage">
      <span class="biomarker-stage-index">01 / Screen</span>
      <div class="biomarker-funnel" role="img" aria-label="Candidate diseases pass through a Sensori screen. Diseases with low movement signal are filtered out, leaving a disease shortlist.">
        <span class="biomarker-funnel-input">Disease candidates</span>
        <span class="biomarker-funnel-particles" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i></span>
        <span class="biomarker-funnel-body"><b>Sensori</b></span>
        <span class="biomarker-funnel-reject"><i aria-hidden="true"></i><small>Low signal</small></span>
        <span class="biomarker-funnel-kept" aria-hidden="true"><i></i><i></i></span>
        <strong class="biomarker-funnel-output">Prioritised</strong>
      </div>
      <b>Is there useful signal?</b>
    </li>
    <li class="biomarker-pathway-stage">
      <span class="biomarker-stage-index">02 / Discover</span>
      <div class="biomarker-funnel" role="img" aria-label="Candidate movement measures are compared with the Sensori benchmark. Measures with weak recovery of the signal are filtered out, leaving a biomarker shortlist.">
        <span class="biomarker-funnel-input">Candidate biomarkers</span>
        <span class="biomarker-funnel-particles" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i></span>
        <span class="biomarker-funnel-body"><b>Benchmark</b></span>
        <span class="biomarker-funnel-reject"><i aria-hidden="true"></i><small>Weak</small></span>
        <span class="biomarker-funnel-kept" aria-hidden="true"><i></i><i></i></span>
        <strong class="biomarker-funnel-output">Shortlist</strong>
      </div>
      <b>Can a simpler measure recover it?</b>
    </li>
    <li class="biomarker-pathway-stage">
      <span class="biomarker-stage-index">03 / Validate</span>
      <div class="biomarker-funnel" role="img" aria-label="Shortlisted biomarkers are tested across devices, populations and time. Unreliable measures are filtered out, leaving a validated digital biomarker.">
        <span class="biomarker-funnel-input">Shortlisted biomarkers</span>
        <span class="biomarker-funnel-particles" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i></span>
        <span class="biomarker-funnel-body"><b>Validation</b></span>
        <span class="biomarker-funnel-reject"><i aria-hidden="true"></i><small>Unreliable</small></span>
        <span class="biomarker-funnel-kept" aria-hidden="true"><i></i><i></i></span>
        <strong class="biomarker-funnel-output">Validated</strong>
      </div>
      <b>Is it robust and clinically useful?</b>
    </li>
  </ol>
</figure>


## The future of movement sensing
Movement sensing is becoming a continuous part of everyday life. Its value should grow as measurements extend over time and are combined with complementary physiological and clinical data. The next challenge is not simply to collect more signals, but to make them accessible to models that can reason across modalities and communicate their findings through language.

There are two promising directions. In the near term, engineered features and structured descriptions of sensor data can provide a practical interface to general-purpose language models; [SensorLM](https://proceedings.neurips.cc/paper_files/paper/2025/hash/42cd98f0e7520d4a63c34891ac1c972f-Abstract-Conference.html) illustrates the broader potential of connecting wearable signals with language. Longer term, movement–language models could align rich representations, such as Sensori embeddings, directly with text. Much as vision–language models connect images with language, this approach could make continuous movement easier to query and interpret while preserving more of the information in the original signal.
