```markdown
# Fine-tune a Pretrained Model

## Open In Colab
## Open In Studio Lab

There are significant benefits to using a pretrained model. It reduces computation costs, your carbon footprint, and allows you to use state-of-the-art models without having to train one from scratch. 🤗 Transformers provides access to thousands of pretrained models for a wide range of tasks. When you use a pretrained model, you train it on a dataset specific to your task. This is known as fine-tuning, an incredibly powerful training technique. In this tutorial, you will fine-tune a pretrained model with a deep learning framework of your choice:

1. Fine-tune a pretrained model with 🤗 Transformers Trainer.
2. Fine-tune a pretrained model in TensorFlow with Keras.
3. Fine-tune a pretrained model in native PyTorch.

## Prepare a Dataset

Before you can fine-tune a pretrained model, download a dataset and prepare it for training. The previous tutorial showed you how to process data for training, and now you get an opportunity to put those skills to the test!

Begin by loading the Yelp Reviews dataset:

```python
from datasets import load_dataset

dataset = load_dataset("yelp_review_full")
dataset["train"][100]
# Output:
# {'label': 0,
#  'text': 'My expectations for McDonalds are t rarely high. But for one to still fail so spectacularly...that takes something special!\\nThe cashier took my friends\'s order, then promptly ignored me. I had to force myself in front of a cashier who opened his register to wait on the person BEHIND me. I waited over five minutes for a gigantic order that included precisely one kid\'s meal. After watching two people who ordered after me be handed their food, I asked where mine was. The manager started yelling at the cashiers for \\"serving off their orders\\" when they didn\'t have their food. But neither cashier was anywhere near those controls, and the manager was the one serving food to customers and clearing the boards.\\nThe manager was rude when giving me my order. She didn\'t make sure that I had everything ON MY RECEIPT, and never even had the decency to apologize that I felt I was getting poor service.\\nI\'ve eaten at various McDonalds restaurants for over 30 years. I\'ve worked at more than one location. I expect bad days, bad moods, and the occasional mistake. But I have yet to have a decent experience at this store. It will remain a place I avoid unless someone in my party needs to avoid illness from low blood sugar. Perhaps I should go back to the racially biased service of Steak n Shake instead!'}
```

As you now know, you need a tokenizer to process the text and include a padding and truncation strategy to handle any variable sequence lengths. To process your dataset in one step, use 🤗 Datasets map method to apply a preprocessing function over the entire dataset:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

If you like, you can create a smaller subset of the full dataset to fine-tune on to reduce the time it takes:

```python
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

## Train

### PyTorch

#### Train with PyTorch Trainer

🤗 Transformers provides a `Trainer` class optimized for training 🤗 Transformers models, making it easier to start training without manually writing your own training loop. The `Trainer` API supports a wide range of training options and features such as logging, gradient accumulation, and mixed precision.

Start by loading your model and specify the number of expected labels. From the Yelp Review dataset card, you know there are five labels:

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
```

You will see a warning about some of the pretrained weights not being used and some weights being randomly initialized. Don’t worry, this is completely normal! The pretrained head of the BERT model is discarded, and replaced with a randomly initialized classification head. You will fine-tune this new model head on your sequence classification task, transferring the knowledge of the pretrained model to it.

#### Training Hyperparameters

Next, create a `TrainingArguments` class which contains all the hyperparameters you can tune as well as flags for activating different training options. For this tutorial you can start with the default training hyperparameters, but feel free to experiment with these to find your optimal settings.

Specify where to save the checkpoints from your training:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="test_trainer")
```

#### Evaluate

`Trainer` does not automatically evaluate model performance during training. You’ll need to pass `Trainer` a function to compute and report metrics. The 🤗 Evaluate library provides a simple accuracy function you can load with the `evaluate.load` (see this quicktour for more information) function:

```python
import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

If you’d like to monitor your evaluation metrics during fine-tuning, specify the `eval_strategy` parameter in your training arguments to report the evaluation metric at the end of each epoch:

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")
```

#### Trainer

Create a `Trainer` object with your model, training arguments, training and test datasets, and evaluation function:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
```

Then fine-tune your model by calling `train()`:

```python
trainer.train()
```

### TensorFlow

#### Train a TensorFlow Model with Keras

You can also train 🤗 Transformers models in TensorFlow with the Keras API!

##### Loading Data for Keras

When you want to train a 🤗 Transformers model with the Keras API, you need to convert your dataset to a format that Keras understands. If your dataset is small, you can just convert the whole thing to NumPy arrays and pass it to Keras. Let’s try that first before we do anything more complicated.

First, load a dataset. We’ll use the CoLA dataset from the GLUE benchmark, since it’s a simple binary text classification task, and just take the training split for now.

```python
from datasets import load_dataset

dataset = load_dataset("glue", "cola")
dataset = dataset["train"]  # Just take the training split for now
```

Next, load a tokenizer and tokenize the data as NumPy arrays. Note that the labels are already a list of 0 and 1s, so we can just convert that directly to a NumPy array without tokenization!

```python
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
tokenized_data = tokenizer(dataset["sentence"], return_tensors="np", padding=True)
# Tokenizer returns a BatchEncoding, but we convert that to a dict for Keras
tokenized_data = dict(tokenized_data)

labels = np.array(dataset["label"])  # Label is already an array of 0 and 1
```

Finally, load, compile, and fit the model. Note that Transformers models all have a default task-relevant loss function, so you don’t need to specify one unless you want to:

```python
from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam

# Load and compile our model
model = TFAutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased")
# Lower learning rates are often better for fine-tuning transformers
model.compile(optimizer=Adam(3e-5))  # No loss argument!

model.fit(tokenized_data, labels)
```

You don’t have to pass a loss argument to your models when you `compile()` them! Hugging Face models automatically choose a loss that is appropriate for their task and model architecture if this argument is left blank. You can always override this by specifying a loss yourself if you want to!

This approach works great for smaller datasets, but for larger datasets, you might find it starts to become a problem. Why? Because the tokenized array and labels would have to be fully loaded into memory, and because NumPy doesn’t handle “jagged” arrays, so every tokenized sample would have to be padded to the length of the longest sample in the whole dataset. That’s going to make your array even bigger, and all those padding tokens will slow down training too!

##### Loading Data as a `tf.data.Dataset`

If you want to avoid slowing down training, you can load your data as a `tf.data.Dataset` instead. Although you can write your own `tf.data` pipeline if you want, we have two convenience methods for doing this:

1. `prepare_tf_dataset()`: This is the method we recommend in most cases. Because it is a method on your model, it can inspect the model to automatically figure out which columns
