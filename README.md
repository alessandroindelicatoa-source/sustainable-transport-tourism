# Sustainable Transport Systems and Mobility in Tourism

Teaching starter package for November 2026.

## Slides

`main.tex` is the editable LaTeX Beamer source; `main.pdf` is the compiled 43-slide preview.

To build the slides:

```bash
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Streamlit lab

Install Python dependencies and launch the local app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The sidebar provides nine modules, aligned with the nine lessons in the uploaded teaching materials. All numerical defaults are illustrative teaching assumptions, not empirical transport data, externally sourced emission factors or estimates of choice behaviour.

## Suggested next development

Expand each lesson into full teaching slides with citations and worked examples, add lesson-specific question sheets and datasets, and add CSV exports or student answer logging if classroom assessment is intended.
