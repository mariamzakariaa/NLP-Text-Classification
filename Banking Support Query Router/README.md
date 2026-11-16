# Banking Support Query Router

A fine-tuned DistilBERT model that reads a customer's message to a bank's support chat and routes it to one of **77 intents** (for example `card\_arrival`, `exchange\_rate`, `lost\_or\_stolen\_card`). A router like this decides which team or bot flow should handle the message.

- **Model on Hugging Face:** https://huggingface.co/mariamzakariaa/banking-support-query-router
- **Notebook:** `Banking_support_query_router.ipynb` (in this repo)
- **Stack:** Python, PyTorch, Hugging Face Transformers + Datasets, scikit-learn, Gradio

## Results at a glance

Two models were compared: a baseline, and the same model trained with typo augmentation. The saved final model is the **typo-augmented** one.

| Run | Val macro-F1 | Test accuracy | Test precision (macro) | Test recall (macro) | Test macro-F1 | Test weighted-F1 |
|---|---|---|---|---|---|---|
| A1 DistilBERT baseline | 0.9013 | 0.9266 | 0.9294 | 0.9266 | 0.9264 | 0.9263 |
| A3 DistilBERT + typo augmentation | 0.9087 | 0.9298 | 0.9330 | 0.9299 | 0.9299 | 0.9299 |

### Robustness to typos

Real customers make typos, but the clean test set hides that. Both models were evaluated on a **noisy copy of the test set** with random character swaps, drops, and duplications.

| Metric | Baseline on noisy test | Augmented on noisy test |
|---|---|---|
| accuracy | 0.7165 | 0.8626 |
| f1_macro | 0.7215 | 0.8628 |

## Data

- **Dataset:** `PolyAI/banking77`, single English customer queries from the banking domain. See the dataset page for its license and terms.
- **Original size:** 10,003 train / 3,080 test; **77 intents**.
- **Text length:** average about 12 words (min 2, max 79).
- **Class sizes in train:** 35 to 187 examples per intent (mean 130), so the data is moderately imbalanced.
- **Quality checks:** 0 duplicates and 0 empty texts in the raw training set.
- **Final splits:** stratified 90/10 train/validation split of the training data, plus the official test set, after removing any train text that also appears in test.

> DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 10003
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 3080
    })
})

## Pipeline

1. **Data review:** number of classes, class balance, text length, duplicates, empties.
2. **Cleaning (deliberately light):** whitespace normalization and deduplication only. Punctuation, digits, and currency symbols are kept because they carry meaning (`?`, `£`, `€`), and lowercasing is left to the uncased tokenizer.
3. **Leakage check:** train texts that also appear in the test set are removed.
4. **Model:** `distilbert-base-uncased`. It is small and fast, and fits short customer messages well.
5. **Fine-tuning:** Hugging Face `Trainer`, with the setup below.
6. **Evaluation:** accuracy, precision, recall, F1 (macro and weighted).
7. **Improvement experiment:** typo augmentation (below).
8. **Testing on new messages and a Gradio demo.**

### Training setup

| Setting | Value |
|---|---|
| Max sequence length | 64 |
| Batch size | 32 |
| Learning rate | 5e-5 |
| Epochs | up to 6, early stopping with patience 2 |
| Weight decay / warmup | 0.01 / 10% of steps |
| Model selection | best validation macro-F1 |
| Seed | 42 |

### Typo augmentation

Each training message was copied with random typos (about 6% of letters swapped with their neighbor, dropped, or duplicated), and the noisy copies were added to the training set, doubling its size. The noisy test set uses the same noise function with fixed per-example seeds. This is synthetic noise, so it approximates but does not replace real user typos.

## Error analysis

The baseline reaches 0.93 accuracy and 0.93 macro-F1 on 3,079 test messages. Five intents are classified perfectly (F1 1.00): `apple_pay_or_google_pay`, `card_about_to_expire`, `passcode_forgotten`, `verify_source_of_funds`, and `verify_top_up`. The errors are not random: they concentrate on **neighboring intents that share the same topic and differ by one small distinction**.

**Top 3 confused intent pairs**

| Count | True intent | Predicted as | Why they get mixed up |
|---|---|---|---|
| 7 | `fiat_currency_support` | `exchange_via_app` | Both talk about currencies. One asks which currencies are supported, the other asks to exchange them. |
| 6 | `why_verify_identity` | `verify_my_identity` | Both are about identity verification. One asks why it is required, the other how to do it. |
| 6 | `declined_transfer` | `declined_card_payment` | Both are "declined" cases. They differ only in the payment channel (transfer vs. card). |

**Weakest intents (F1):** `fiat_currency_support` 0.79, `balance_not_updated_after_bank_transfer` 0.80, and `pending_transfer` 0.83. The lowest recall is 0.72, for both `declined_transfer` and `fiat_currency_support`.

**Individual errors**

| Message | True intent | Predicted (confidence) |
|---|---|---|
| Why am I being charged more ? | `card_payment_wrong_exchange_rate` | `extra_charge_on_statement` (0.34) |
| the conversion value for my card payments is incorrect. | `card_payment_wrong_exchange_rate` | `reverted_card_payment?` (0.44) |
| The app doesn't show the card I received. | `card_linking` | `card_arrival` (0.86) |
| Can I link another card to my account? | `card_linking` | `getting_spare_card` (0.83) |

Two patterns show up. **Vague messages** carry no clue about the cause ("charged more"), and **paraphrases** avoid the usual keywords ("conversion value" instead of "exchange rate"). Both of those errors have low confidence (0.34 and 0.44), so a confidence threshold that sends uncertain messages to a human agent would catch them. The two `card_linking` errors are confident (0.86 and 0.83), so a threshold alone would not catch everything.
## Testing on new messages

I probed several closely related intents about currencies and exchange to see how the model separates them:

- "What is the exchange rate for euros?"
- "What is the exchange rate?"
- "How much is one euro worth?"
- "I want to exchange euros"
- "Can I exchange EUR to USD?"
- "What currencies are supported?"
- "Do you support euros?"

### Test Results

| Message | Predicted Intent | Confidence | Result |
|---|---|---:|---|
| What is the exchange rate for euros? | exchange_rate | 0.83 | Correct |
| What is the exchange rate? | exchange_rate | 1.00 | Correct |
| How much is one euro worth? | exchange_charge | 0.41 | Incorrect |
| I want to exchange euros | exchange_via_app | 0.52 | Correct* |
| Can I exchange EUR to USD? | fiat_currency_support | 0.98 | Correct* |
| What currencies are supported? | fiat_currency_support | 0.99 | Correct |
| Do you support euros? | fiat_currency_support | 0.98 | Correct |

### Error Analysis

The model incorrectly classified "How much is one euro worth?" as `exchange_charge` instead of `exchange_rate`.

This suggests confusion between currency conversion queries and transaction charge queries, possibly due to overlapping training examples.

Overall, the model performs well on these examples but could benefit from more diverse training examples for similar intents.

*Correctness should be verified against the dataset's actual labels.*

## Limitations

- **Single run per configuration, one seed**, so small differences between runs are not conclusive.
- **No larger-model comparison:** only DistilBERT was trained. A larger model such as `bert-base-uncased` is a natural next step.
- **Synthetic typo noise** is not the same as real user mistakes.
- **Scope:** English, single-turn queries in the banking domain. The model has no "none of the above" class, so unrelated messages will still be forced into one of the 77 intents.
- **Overlapping intents:** several of the 77 intents are close in meaning, so some confusion between them is expected.

## Possible improvements

- Add a confidence threshold, so low-confidence messages go to a human agent.
- Compare a larger model and multiple seeds.
- Add an "out of scope" class trained on unrelated messages.
- Allow more languages.

## Use the model

```python

from transformers import pipeline



clf = pipeline("text-classification",

&#x20;              model="mariamzakariaa/banking-support-query-router", top\_k=3)

print(clf("My card still hasn't arrived after two weeks"))

```

## Run it yourself

1. Open `Banking_support_query_router.ipynb` in Google Colab and enable a GPU.
2. Run all cells in order (setup, data, cleaning, baseline, typo experiment, results, save, demo).
3. The Gradio cell gives a demo where you paste a customer message and see the top 3 predicted intents with confidence scores.

```

.

├── Banking_support_query_router.ipynb

├── results_banking_project.csv

└── README.md

```

---

## ملخص بالعربي

نموذج DistilBERT تم عمل Fine-tuning له لتصنيف رسائل دعم عملاء البنك إلى 77 فئة (Intent). المشروع يشمل مراجعة البيانات وتنظيفها، التدريب والتقييم (Accuracy / Precision / Recall / F1)، وتجربة تحسين لمعالجة الأخطاء الإملائية بالـ Data Augmentation، واختبار على رسائل جديدة، وDemo تفاعلي.
