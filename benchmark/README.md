# Codebase Benchmark Tool

This tool is designed to benchmark and evaluate the performance of RAG (Retrieval-Augmented Generation) systems on codebases. It provides a comprehensive suite for creating datasets, running benchmarks, and evaluating the results.

## Features

1. Dataset Creation
2. Benchmarking
3. Context Evaluation
4. Answer Evaluation
5. Detailed Results Analysis

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

Create a `.env` file and add your OpenAI API key:

```bash
OPENAI_API_KEY=<your_openai_api_key>
```

## Usage

### 1. Dataset Creation

To create a dataset from a codebase:


```bash
python create_dataset.py
```


### 2. Running the Benchmark

To run the benchmark:

```bash
python run_benchmark.py
```


### 3. Configuration

The benchmark can be configured using the `benchmark_config.yaml` file.

## Output

The benchmark results are saved in a JSON file, as specified in the configuration. The results include detailed scores for context evaluation and answer quality.

## Customization

You can customize the benchmark details by modifying the question templates in `questions_template.json` and the evaluation prompts in the `utils/llm_interface.py` file.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions to improve the benchmark tool are welcome. Please feel free to submit pull requests or open issues for any bugs or feature requests.
