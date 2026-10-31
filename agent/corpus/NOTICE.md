# Corpus source and license

`passages.jsonl` and `cuad_annotations.jsonl` are derived from the
**Contract Understanding Atticus Dataset (CUAD) v1**, by The Atticus Project,
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- Dataset: https://www.atticusprojectai.org/cuad
- Paper: Hendrycks, Burns, Chen, Ball. *CUAD: An Expert-Annotated NLP Dataset
  for Legal Contract Review*. NeurIPS 2021 Datasets and Benchmarks.

Changes made: the 510 contracts were split into passages of about 1,000
characters with whitespace normalized and heuristic section labels, assigned
IDs `CUAD-0001` … `CUAD-0510` in sorted-title order, and CUAD's clause
annotations were mapped to those passage IDs. Regenerate with:

```bash
python -m agent.ingest_cuad path/to/CUADv1.json
```
