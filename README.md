# LLM Exams Evaluation

This repository accompanies the paper *"Pass or Fail? Lessons from Evaluating LLMs on Medical Exams"*. It documents a systematic evaluation of multilingual and on-premise large language models (LLMs) applied to curated German and Portuguese medical exam datasets.

We evaluate several models (e.g., LLaMA, Mistral, Qwen2.5, Gemma, and GPT-4o) and facilitated the use of retrieval-augmented generation (RAG).

Note: needs instance or connection to server running Ollama with models under analysis properly installed.

> ⚠️ The exam datasets used in this study are not publicly released due to privacy and copyright concerns. However, the results, scripts, and evaluation framework are provided to support reproducibility and further exploration.

## Features

- Evaluation of over 1,500 curated medical questions (German & Portuguese)
- Support for prompt language comparison
- Retrieval-Augmented Generation (RAG) integration
- Runtime performance tracking
- On-premise execution for privacy-compliant benchmarking

## Installation

Clone the project and create the Python environment using the provided `environment.yml` file:

```bash
git clone https://github.com/your-org/llm-exams-evaluation.git
cd llm-exams-evaluation
conda env create -f environment.yml
conda llm-exams-evaluation
```

Create `.env` file with target server running Ollama `OLLAMA_HOST=''` or OpenAI API key `OPENAI_API_KEY=''`.

Fill `curated/` folder with target exams using `exam_template.json` as a template


### RAG Setup

To enable Retrieval-Augmented Generation - Additional setup may be required for manually installing required packages.

## Project Structure

```
├── images/                 # Image folder containing images generated during the results analysis
├── results/                # Results after running model on target exams
├── utils/                  # Auxiliar code
├── .env                    # Environment file
├── environment.yml         # Conda environment specification
├── exam_template.json      # Template with expected format for creating new exam entries
├── rag_test.py             # Run RAG technique
├── README.md               # This file
├── reporter.py             # Analyse results, create plots and tables
├── runner.py               # Main runner, runs exmas using specified models
```

## Hardware

Experiments were run on a local server equipped with an NVIDIA GA100 GPU. GPT-4o was accessed via OpenAI’s API.

## Citation

If you use this work, please cite:

Macedo M., Händel C., Bueno A., Schreweis B., Saalfeld S., Ulrich H. *Pass or Fail? Lessons from Evaluating LLMs on Medical Exams*, 2025.

