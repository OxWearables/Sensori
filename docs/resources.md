---
layout: listing
title: Resources
eyebrow: Paper, code and model
intro: Everything needed to read, reproduce and build on Sensori, collected in one place.
permalink: /resources/
---
<div class="resource-grid">
  <article>
    <span>01</span><h2>Paper</h2><p>The manuscript describing the model, evaluation and implications for passive health monitoring.</p>
    {% if site.links.paper != "" %}<a class="text-link" href="{{ site.links.paper }}">Read the paper <span aria-hidden="true">↗</span></a>{% else %}<p class="status-pill">Available at release</p>{% endif %}
  </article>
  <article>
    <span>02</span><h2>Code</h2><p>Preprocessing, model inference and downstream evaluation code for research use.</p>
    {% if site.links.code != "" %}<a class="text-link" href="{{ site.links.code }}">Browse the code <span aria-hidden="true">↗</span></a>{% else %}<p class="status-pill">Available at release</p>{% endif %}
  </article>
  <article>
    <span>03</span><h2>Model</h2><p>Pretrained weights and documentation for generating minute- and day-level representations.</p>
    {% if site.links.model != "" %}<a class="text-link" href="{{ site.links.model }}">Download weights <span aria-hidden="true">↗</span></a>{% else %}<p class="status-pill">Available at release</p>{% endif %}
  </article>
</div>

<section class="citation-block" id="citation">
  <div><p class="eyebrow">Citation</p><h2>Cite Sensori</h2></div>
  <div><p>If you find this paper or code useful in your research, please consider citing our <a href="https://arxiv.org/abs/2608.29494">paper</a>.</p><pre><code>@misc{wang2026learning,
  title         = {Learning Human Health and Diseases from 24-hour Wrist Movement},
  author        = {Wang, Yong and McGagh, Dylan and Broomberg, Katya and Zhang, Zizheng and Carter, Jonathan and Naushad, Junayed and Brocklebank, Laura and Sun, Yang and Nicholson, George and Sun, Dianjianyi and Yu, Canqing and Lv, Jun and Barnard, Maxim and Lam, Hubert and Steptoe, Andrew and Eyre, David W. and Li, Liming and Chen, Zhengming and Wray, Naomi and Denaxas, Spiros and Collins, Gary S. and Du, Huaidong and Doherty, Aiden and Yuan, Hang},
  year          = {2026},
  eprint        = {2608.29494},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2608.29494}
}</code></pre></div>
</section>

<section class="data-access">
  <p class="eyebrow">Data access</p>
  <h2>Built across four population studies.</h2>
  <p>UK Biobank and China Kadoorie Biobank data are available to approved researchers through their respective access platforms. ELSA is available through the UK Data Service; NHANES and the human activity recognition benchmarks are publicly available from their original sources.</p>
</section>
