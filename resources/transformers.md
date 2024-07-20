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
