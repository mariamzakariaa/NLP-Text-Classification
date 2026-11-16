# Arabic Dialect Identifier

A fine-tuned MARBERT model that reads an Arabic text and predicts its dialect: **Egyptian, Gulf, Levantine, Maghrebi, or Modern Standard Arabic (MSA)**.

- **Model on Hugging Face:** https://huggingface.co/mariamzakariaa/arabic-dialect-identifier
- **Notebook:** `arabic-dialect-identifier.ipynb` (in this repo)
- **Stack:** Python, PyTorch, Hugging Face Transformers + Datasets, scikit-learn, Gradio

## Results at a glance

Three runs were compared. The final model is **B2 (MARBERT + class weights)**, chosen by validation macro-F1.

| Run | Val macro-F1 | Test accuracy | Test precision (macro) | Test recall (macro) | Test macro-F1 | Test weighted-F1 |
|---|---|---|---|---|---|---|
| B1 MARBERT baseline | 0.5222 | **0.7975** | 0.5211 | 0.4788 | 0.4924 | **0.7882** |
| **B2 MARBERT + class weights** | **0.6375** | 0.7670 | **0.5327** | **0.5647** | **0.5327** | 0.7733 |
| B3 CAMeLBERT-DA + class weights | 0.5949 | 0.7521 | 0.4818 | 0.5555 | 0.5095 | 0.7614 |

**How to read this:** the baseline has the highest accuracy because the data is dominated by MSA, and predicting MSA well is enough to score high. Class weights trade about 3 points of accuracy (0.7975 → 0.7670) for about +4 points of macro-F1 (0.4924 → 0.5327) and +8.6 points of macro-recall (0.4788 → 0.5647), meaning the rarer dialects are detected more often. Macro-F1 is the headline metric because the classes are heavily imbalanced.

## Data

- **Dataset:** ArSarcasm (`iabufarha/ar\_sarcasm`), Arabic tweets with a dialect label. Only the `tweet` and `dialect` columns are used. See the dataset page for its license and terms.
- **Original size:** 8,437 train / 2,110 test tweets.
- **After cleaning:** 7,233 train / 804 validation / 2,094 test.
  - Short or duplicate tweets removed: 268 from train, 16 from test.
  - Another 132 train tweets removed because the same text also appears in the test set (to avoid leakage).
  - Validation is a stratified 10% split of the training data.
- **Class distribution (original train):**

| Dialect | Tweets | Share | Test support |
|---|---|---|---|
| MSA | 5,652 | 67.0% | 1,397 |
| Egyptian | 1,904 | 22.6% | 477 |
| Levantine | 439 | 5.2% | 112 |
| Gulf | 414 | 4.9% | 104 |
| Maghrebi | 28 | 0.3% | **4** |

Average tweet length is about 15 words.

## Pipeline

1. **Data review:** class balance, text lengths, duplicates. The main finding is the strong imbalance, especially Maghrebi with only 28 training tweets.
2. **Cleaning (deliberately light):**
   - Removed links, replaced mentions with `@user`, removed `#` and `\_`, removed diacritics and tatweel, and shortened letters repeated 3+ times (e.g. حلوووو → حلوو).
   - **Not done on purpose:** no letter normalization (alef/yaa/taa marbuta) and no stopword removal, because spelling variants and small function words ("إيه", "شو", "وش") are exactly the dialect signals the model needs.
3. **Models:** `UBC-NLP/MARBERT` (pre-trained on Arabic tweets, including dialects) as the baseline, and `CAMeL-Lab/bert-base-arabic-camelbert-da` (dialect-adapted) as a comparison.
4. **Fine-tuning:** Hugging Face `Trainer`, with the setup below.
5. **Evaluation:** accuracy, precision, recall, F1 (macro and weighted), a per-class report, and a confusion matrix.
6. **Improvements tried:** class-weighted loss, then a different pre-trained model.

### Training setup

| Setting | Value |
|---|---|
| Max sequence length | 128 |
| Batch size | 32 |
| Learning rate | 2e-5 (MARBERT), 3e-5 (CAMeLBERT-DA) |
| Epochs | up to 5, early stopping with patience 2 |
| Weight decay / warmup | 0.01 / 10% of steps |
| Model selection | best validation macro-F1 |
| Seed | 42 |
| Class weights (B2, B3) | "balanced": egypt 0.88, gulf 4.53, levant 3.83, magreb 68.89, msa 0.30 |
| Time per run | about 2.5 to 4 minutes on a GPU |

Best epochs: B1 at epoch 3, B2 at epoch 1 (training stopped after epoch 3), B3 at epoch 4.

## Per-class results (B1 baseline, test set)

| Dialect | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Egyptian | 0.74 | 0.62 | 0.68 | 477 |
| Gulf | 0.50 | 0.56 | 0.53 | 104 |
| Levantine | 0.52 | 0.29 | 0.38 | 112 |
| Maghrebi | 0.00 | 0.00 | 0.00 | 4 |
| MSA | 0.85 | 0.92 | 0.88 | 1,397 |

| Dialect | B1 Precision | B1 Recall | B1 F1 | B2 Precision | B2 Recall | B2 F1 | Support |
|---|---:|---:|---:|---:|---:|---:|---:|
| Egyptian | 0.74 | 0.62 | 0.68 | 0.72 | 0.62 | 0.67 | 477 |
| Gulf | 0.50 | 0.56 | 0.53 | 0.36 | 0.75 | 0.48 | 104 |
| Levantine | 0.52 | 0.29 | 0.38 | 0.38 | 0.35 | 0.36 | 112 |
| Maghrebi | 0.00 | 0.00 | 0.00 | 0.33 | 0.25 | 0.29 | 4 |
| MSA | 0.85 | 0.92 | 0.88 | 0.88 | 0.85 | 0.87 | 1,397 |

B2 showed a slight improvement in Egyptian precision but unchanged recall, a decrease in Gulf precision with improved recall, a small decline in Levantine performance, improved Maghrebi scores from zero, and a slight decrease in MSA performance overall.

## Error analysis

- **MSA dominates.** Without class weights the baseline is biased toward the majority class: MSA F1 is 0.88, while Levantine recall is only 0.29 and Maghrebi is never predicted correctly.
- **Maghrebi is not reliably learnable here.** There are only 28 training tweets and 4 test tweets, so its scores are extremely noisy. A class weight of 68.89 on 28 examples is also unstable. My own Maghrebi test sentence (below) was misclassified as Levantine.
- **Some "errors" look like label noise.** Several misclassified examples have labels that look questionable, for instance:

| Tweet (shortened) | Label | Predicted |
|---|---|---|
| جوجل تنشر قائمة أجهزة نيكسس التي ستحصل على تحديث أندرويد 7.1 | Egyptian | MSA (0.95) |
| بعدك على بالي 😄 فيروز | Egyptian | MSA (0.42) |

  The first is a plain news-style sentence, and the second contains the Levantine word "بعدك". Dialect labels on tweets are hard to assign, so the achievable ceiling on this dataset is probably well below 100%.
- **Short texts are ambiguous.** Many short tweets contain few dialect markers and share vocabulary with MSA or neighboring dialects.

## Testing on new sentences (final model B2)

| Sentence | Expected | Top prediction | Correct? |
|---|---|---|---|
| إنت بتعمل إيه دلوقتي؟ تعالى نشرب قهوة | Egyptian | Egyptian (0.80) | Yes |
| شلونك وش تسوي الحين؟ | Gulf | Gulf (0.61) | Yes |
| كيفك شو عم تعمل هلق؟ | Levantine | Levantine (0.59) | Yes |
| شنو كتدير دابا؟ | Maghrebi | Levantine (0.30) | **No** (Maghrebi not in top 3) |
| ماذا تفعل الآن؟ لقد انتظرتك طويلا | MSA | MSA (0.44), Egyptian 0.34 | Yes, but low confidence |

The strongest dialects (Egyptian, Gulf, Levantine) work well on clear sentences. Maghrebi fails, consistent with the tiny amount of training data. Low confidence scores (below about 0.5) signal an unreliable prediction.

<p align="center">
  <img src="arabic-dialect%20(1).png" width="500" alt="Demo 1">
  <img src="arabic-dialect%20(2).png" width="500" alt="Demo 2">
  <img src="arabic-dialect%20(3).png" width="500" alt="Demo 3">
</p>

## Limitations

- **Single run per configuration, one seed, no confidence intervals.** The B2 vs B3 gap (0.5327 vs 0.5095 test macro-F1) is small, so don't read it as a firm ranking. B2 vs the baseline on macro-recall is a larger gap.
- **Validation is small** (804 tweets, with only a handful of Maghrebi), so validation macro-F1 is noisy.
- **Domain:** trained on tweets. Other kinds of text (formal writing, speech transcripts) may behave differently.
- **Labels may be noisy** (see error analysis), and a dialect is not a reliable indicator of where a person is from. This model is a demo, not a tool for profiling people.

## Possible improvements

- More Maghrebi data, or a more balanced dialect dataset (e.g. NADI), would be the biggest single gain.
- Softer class weights (for example the square root of the balanced weights) to avoid the extreme Maghrebi weight.
- Multiple seeds to report mean and variance.
- Show a "low confidence" warning in the demo when the top score is below 0.5.

## Use the model

```python

from transformers import pipeline



clf = pipeline("text-classification",

&#x20;              model="mariamzakariaa/arabic-dialect-identifier", top\_k=3)

print(clf("كيفك شو عم تعمل هلق؟"))

```

## Run it yourself

1. Open `project\_B\_arabic\_dialect.ipynb` in Google Colab or Kaggle and enable a GPU.
2. Run all cells in order (setup, data, cleaning, B1, B2, B3, evaluation, save, demo).
3. The last cell runs a Gradio demo where you can paste any Arabic text and see the top 3 dialects with confidence scores.

```

.

├── arabic-dialect-identifier.ipynb

├── results\_project\_B.csv

├── confusion\_matrix\_B1.png

└── README.md

```

---

## ملخص بالعربي

نموذج MARBERT تم عمل Fine-tuning له لتحديد اللهجة العربية من النص (مصري / خليجي / شامي / مغاربي / فصحى). أفضل تجربة كانت باستخدام Class Weights وحققت Macro-F1 = 0.53 ودقة 77%. أضعف فئة هي المغاربي بسبب قلة البيانات (28 تغريدة فقط في التدريب). المشروع يشمل تنظيف النصوص، مقارنة بين نموذجين، تحليل الأخطاء، واختبار على جمل جديدة.
