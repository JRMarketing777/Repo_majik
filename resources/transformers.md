```markdown
# Quick Tour

## Open In Colab

## Open In Studio Lab

Get up and running with 🤗 Transformers! Whether you’re a developer or an everyday user, this quick tour will help you get started and show you how to use the `pipeline()` for inference, load a pretrained model and preprocessor with an AutoClass, and quickly train a model with PyTorch or TensorFlow. If you’re a beginner, we recommend checking out our tutorials or course next for more in-depth explanations of the concepts introduced here.

Before you begin, make sure you have all the necessary libraries installed:

```bash
!pip install transformers datasets evaluate accelerate
```

You’ll also need to install your preferred machine learning framework:

### PyTorch

```bash
pip install torch
```

### TensorFlow

```bash
pip install tensorflow
```

## Pipeline

The `pipeline()` is the easiest and fastest way to use a pretrained model for inference. You can use the `pipeline()` out-of-the-box for many tasks across different modalities, some of which are shown in the table below:

| Task                         | Description                                            | Modality         | Pipeline identifier                  |
|------------------------------|--------------------------------------------------------|------------------|--------------------------------------|
| Text classification          | assign a label to a given sequence of text             | NLP              | `pipeline(task="sentiment-analysis")`|
| Text generation              | generate text given a prompt                           | NLP              | `pipeline(task="text-generation")`   |
| Summarization                | generate a summary of a sequence of text or document   | NLP              | `pipeline(task="summarization")`     |
| Image classification         | assign a label to an image                             | Computer vision  | `pipeline(task="image-classification")`|
| Image segmentation           | assign a label to each individual pixel of an image    | Computer vision  | `pipeline(task="image-segmentation")`|
| Object detection             | predict the bounding boxes and classes of objects      | Computer vision  | `pipeline(task="object-detection")`  |
| Audio classification         | assign a label to some audio data                      | Audio            | `pipeline(task="audio-classification")`|
| Automatic speech recognition | transcribe speech into text                            | Audio            | `pipeline(task="automatic-speech-recognition")`|
| Visual question answering    | answer a question about the image, given an image and a question | Multimodal      | `pipeline(task="vqa")`               |
| Document question answering  | answer a question about the document, given a document and a question | Multimodal | `pipeline(task="document-question-answering")`|
| Image captioning             | generate a caption for a given image                   | Multimodal       | `pipeline(task="image-to-text")`     |

For a complete list of available tasks, check out the [pipeline API reference](https://huggingface.co/transformers/main_classes/pipelines.html).

Start by creating an instance of `pipeline()` and specifying a task you want to use it for. In this guide, you’ll use the `pipeline()` for sentiment analysis as an example:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
```

The `pipeline()` downloads and caches a default pretrained model and tokenizer for sentiment analysis. Now you can use the classifier on your target text:

```python
classifier("We are very happy to show you the 🤗 Transformers library.")
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

If you have more than one input, pass your inputs as a list to the `pipeline()` to return a list of dictionaries:

```python
results = classifier([
    "We are very happy to show you the 🤗 Transformers library.",
    "We hope you don't hate it."
])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
# label: POSITIVE, with score: 0.9998
# label: NEGATIVE, with score: 0.5309
```

The `pipeline()` can also iterate over an entire dataset for any task you like. For this example, let’s choose automatic speech recognition as our task:

```python
import torch
from transformers import pipeline

speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
```

Load an audio dataset (see the [🤗 Datasets Quick Start](https://huggingface.co/docs/datasets/quicktour.html) for more details) you’d like to iterate over. For example, load the MInDS-14 dataset:

```python
from datasets import load_dataset, Audio

dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
```

You need to make sure the sampling rate of the dataset matches the sampling rate `facebook/wav2vec2-base-960h` was trained on:

```python
dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))
```

The audio files are automatically loaded and resampled when calling the "audio" column. Extract the raw waveform arrays from the first 4 samples and pass it as a list to the pipeline:

```python
result = speech_recognizer(dataset[:4]["audio"])
print([d["text"] for d in result])
# ['I WOULD LIKE TO SET UP A JOINT ACCOUNT WITH MY PARTNER HOW DO I PROCEED WITH DOING THAT',
# "FONDERING HOW I'D SET UP A JOIN TO HELL T WITH MY WIFE AND WHERE THE AP MIGHT BE",
# "I I'D LIKE TOY SET UP A JOINT ACCOUNT WITH MY PARTNER I'M NOT SEEING THE OPTION TO DO IT ON THE APSO I CALLED IN TO GET SOME HELP CAN I JUST DO IT OVER THE PHONE WITH YOU AND GIVE YOU THE INFORMATION OR SHOULD I DO IT IN THE AP AN I'M MISSING SOMETHING UQUETTE HAD PREFERRED TO JUST DO IT OVER THE PHONE OF POSSIBLE THINGS",
# 'HOW DO I FURN A JOINA COUT']
```

For larger datasets where the inputs are big (like in speech or vision), you’ll want to pass a generator instead of a list to load all the inputs in memory. Take a look at the [pipeline API reference](https://huggingface.co/transformers/main_classes/pipelines.html) for more information.

## Use Another Model and Tokenizer in the Pipeline

The `pipeline()` can accommodate any model from the Hub, making it easy to adapt the `pipeline()` for other use-cases. For example, if you’d like a model capable of handling French text, use the tags on the Hub to filter for an appropriate model. The top filtered result returns a multilingual BERT model finetuned for sentiment analysis you can use for French text:

```python
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
```

### PyTorch

Use `AutoModelForSequenceClassification` and `AutoTokenizer` to load the pretrained model and its associated tokenizer (more on an AutoClass in the next section):

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### TensorFlow

Use `TFAutoModelForSequenceClassification` and `AutoTokenizer` to load the pretrained model and its associated tokenizer (more on a TFAutoClass in the next section):

```python
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

Specify the model and tokenizer in the `pipeline()`, and now you can apply the classifier on French text:

```python
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.")
# [{'label': '5 stars', 'score': 0.7273}]
```

If you can’t find a model for your use-case, you’ll need to finetune a pretrained model on your data. Take a look at our [finetuning tutorial](https://huggingface.co/transformers/training.html) to learn how. Finally, after you’ve finetuned your pretrained model, please consider sharing the model with the community on the Hub to democratize machine learning for everyone! 🤗

## AutoClass

Under the hood, the `AutoModelForSequenceClassification` and `AutoTokenizer` classes work together to power the `pipeline()` you used above. An AutoClass is a shortcut that automatically retrieves the architecture of a pretrained model from its name or path. You only need to select the appropriate AutoClass for your task and its associated preprocessing class.

Let’s return to the example from the previous section and see how you can use the AutoClass to replicate the results of the `pipeline()`.

### AutoTokenizer

A tokenizer is responsible for preprocessing text into an array of numbers as inputs to a model. There are multiple rules that govern the tokenization process, including how to split a word and at what level words should be split (learn more about tokenization in the [tokenizer summary](https://huggingface.co/transformers/tokenizer_summary.html)). The most important thing to remember is you need to instantiate a tokenizer with the same model name to ensure you’re using the same tokenization rules a model was pretrained with.

Load a tokenizer with `AutoTokenizer`:

```python
from transformers import AutoTokenizer

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
token

Sure! Here's the markdown version of the installation instructions for Hugging Face's Transformers library:

```markdown
# Installation

Install 🤗 Transformers for whichever deep learning library you’re working with, set up your cache, and optionally configure 🤗 Transformers to run offline.

🤗 Transformers is tested on Python 3.6+, PyTorch 1.1.0+, TensorFlow 2.0+, and Flax. Follow the installation instructions below for the deep learning library you are using:

- [PyTorch installation instructions](https://pytorch.org/get-started/locally/).
- [TensorFlow 2.0 installation instructions](https://www.tensorflow.org/install).
- [Flax installation instructions](https://flax.readthedocs.io/en/latest/installation.html).

## Install with pip

You should install 🤗 Transformers in a virtual environment. If you’re unfamiliar with Python virtual environments, take a look at [this guide](https://realpython.com/python-virtual-environments-a-primer/). A virtual environment makes it easier to manage different projects and avoid compatibility issues between dependencies.

Start by creating a virtual environment in your project directory:

```bash
python -m venv .env
```

Activate the virtual environment. On Linux and MacOS:

```bash
source .env/bin/activate
```

Activate the virtual environment on Windows:

```bash
.env/Scripts/activate
```

Now you’re ready to install 🤗 Transformers with the following command:

```bash
pip install transformers
```

For CPU-support only, you can conveniently install 🤗 Transformers and a deep learning library in one line. For example, install 🤗 Transformers and PyTorch with:

```bash
pip install 'transformers[torch]'
```

🤗 Transformers and TensorFlow 2.0:

```bash
pip install 'transformers[tf-cpu]'
```

### M1 / ARM Users

You will need to install the following before installing TensorFlow 2.0:

```bash
brew install cmake
brew install pkg-config
```

🤗 Transformers and Flax:

```bash
pip install 'transformers[flax]'
```

Finally, check if 🤗 Transformers has been properly installed by running the following command. It will download a pretrained model:

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

Then print out the label and score:

```bash
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

## Install from source

Install 🤗 Transformers from source with the following command:

```bash
pip install git+https://github.com/huggingface/transformers
```

This command installs the bleeding edge main version rather than the latest stable version. The main version is useful for staying up-to-date with the latest developments. For instance, if a bug has been fixed since the last official release but a new release hasn’t been rolled out yet. However, this means the main version may not always be stable. We strive to keep the main version operational, and most issues are usually resolved within a few hours or a day. If you run into a problem, please open an Issue so we can fix it even sooner!

Check if 🤗 Transformers has been properly installed by running the following command:

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I love you'))"
```

## Editable install

You will need an editable install if you’d like to:

- Use the main version of the source code.
- Contribute to 🤗 Transformers and need to test changes in the code.

Clone the repository and install 🤗 Transformers with the following commands:

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

These commands will link the folder you cloned the repository to and your Python library paths. Python will now look inside the folder you cloned to in addition to the normal library paths. For example, if your Python packages are typically installed in `~/anaconda3/envs/main/lib/python3.7/site-packages/`, Python will also search the folder you cloned to: `~/transformers/`.

You must keep the transformers folder if you want to keep using the library.

Now you can easily update your clone to the latest version of 🤗 Transformers with the following command:

```bash
cd ~/transformers/
git pull
```

Your Python environment will find the main version of 🤗 Transformers on the next run.

## Install with conda

Install from the conda channel conda-forge:

```bash
conda install conda-forge::transformers
```

## Cache setup

Pretrained models are downloaded and locally cached at: `~/.cache/huggingface/hub`. This is the default directory given by the shell environment variable `TRANSFORMERS_CACHE`. On Windows, the default directory is given by `C:\Users\username\.cache\huggingface\hub`. You can change the shell environment variables shown below - in order of priority - to specify a different cache directory:

- Shell environment variable (default): `HUGGINGFACE_HUB_CACHE` or `TRANSFORMERS_CACHE`.
- Shell environment variable: `HF_HOME`.
- Shell environment variable: `XDG_CACHE_HOME` + `/huggingface`.

🤗 Transformers will use the shell environment variables `PYTORCH_TRANSFORMERS_CACHE` or `PYTORCH_PRETRAINED_BERT_CACHE` if you are coming from an earlier iteration of this library and have set those environment variables, unless you specify the shell environment variable `TRANSFORMERS_CACHE`.

## Offline mode

Run 🤗 Transformers in a firewalled or offline environment with locally cached files by setting the environment variable `HF_HUB_OFFLINE=1`.

Add 🤗 Datasets to your offline training workflow with the environment variable `HF_DATASETS_OFFLINE=1`.

```bash
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python examples/pytorch/translation/run_translation.py --model_name_or_path google/t5-small --dataset_name wmt16 --dataset_config ro-en ...
```

This script should run without hanging or waiting to timeout because it won’t attempt to download the model from the Hub.

You can also bypass loading a model from the Hub from each `from_pretrained()` call with the `local_files_only` parameter. When set to `True`, only local files are loaded:

```python
from transformers import T5Model

model = T5Model.from_pretrained("./path/to/local/directory", local_files_only=True)
```

## Fetch models and tokenizers to use offline

Another option for using 🤗 Transformers offline is to download the files ahead of time, and then point to their local path when you need to use them offline. There are three ways to do this:

1. Download a file through the user interface on the Model Hub by clicking on the ↓ icon.

2. Use the `PreTrainedModel.from_pretrained()` and `PreTrainedModel.save_pretrained()` workflow:

   Download your files ahead of time with `PreTrainedModel.from_pretrained()`:

   ```python
   from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

   tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
   model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")
   ```

   Save your files to a specified directory with `PreTrainedModel.save_pretrained()`:

   ```python
   tokenizer.save_pretrained("./your/path/bigscience_t0")
   model.save_pretrained("./your/path/bigscience_t0")
   ```

   Now when you’re offline, reload your files with `PreTrainedModel.from_pretrained()` from the specified directory:

   ```python
   tokenizer = AutoTokenizer.from_pretrained("./your/path/bigscience_t0")
   model = AutoModel.from_pretrained("./your/path/bigscience_t0")
   ```

3. Programmatically download files with the `huggingface_hub` library:

   Install the `huggingface_hub` library in your virtual environment:

   ```bash
   python -m pip install huggingface_hub
   ```

   Use the `hf_hub_download` function to download a file to a specific path. For example, the following command downloads the `config.json` file from the T0 model to your desired path:

   ```python
   from huggingface_hub import hf_hub_download

   hf_hub_download(repo_id="bigscience/T0_3B", filename="config.json", cache_dir="./your/path/bigscience_t0")
   ```

   Once your file is downloaded and locally cached, specify its local path to load and use it:

   ```python
   from transformers import AutoConfig

   config = AutoConfig.from_pretrained("./your/path/bigscience_t0/config.json")
   ```

See the [How to download files from the Hub](https://huggingface.co/docs/hub/main/guides/download) section for more details on downloading files stored on the Hub.
```

This markdown document provides a comprehensive guide on installing and using the Hugging Face Transformers library, including setting up a virtual environment, installing dependencies, configuring cache, and running in offline mode.
