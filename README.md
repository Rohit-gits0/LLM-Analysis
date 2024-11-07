# LLM-Analysis

# Serverless Language Translation Models Comparison

A comprehensive comparison of state-of-the-art language translation models (mBART-50, M2M-100, NLLB-200, and Google Translate) for Hindi-English translation using Hugging Face's cloud architecture.

## 📊 Key Findings

### English to Hindi Translation

#### IIT-B Dataset Results
| Model           | BLEU | STS  | chrF | METEOR |
|-----------------|------|------|------|--------|
| NLLB-200        | 42.79 | 78.28 | 36.29 | 27.73 |
| M2M100          | 36.36 | 72.86 | 28.98 | 20.59 |
| mBART-50        | 45.07 | 77.10 | 37.77 | 32.50 |
| Google Translator | 42.57 | 76.24 | 35.64 | 22.87 |
| IndicTrans      | 49.71 | 77.81 | 56.62 | 31.81 |

#### Bhagavad Gita Dataset Results
| Model           | BLEU | STS  | chrF | METEOR |
|-----------------|------|------|------|--------|
| NLLB-200        | 39.71 | 52.57 | 46.70 | 23.21 |
| M2M100          | 30.83 | 63.24 | 34.30 | 29.48 |
| mBART-50        | 40.38 | 52.52 | 45.46 | 22.51 |
| Google Translator | 83.15 | 82.30 | 81.20 | 78.75 |
| IndicTrans      | 42.75 | 54.71 | 52.49 | 25.48 |

### Hindi to English Translation

#### IIT-B Dataset Results
| Model           | BLEU | STS  | chrF | METEOR |
|-----------------|------|------|------|--------|
| NLLB-200        | 56.64 | 79.55 | 55.47 | 46.65 |
| M2M100          | 45.53 | 68.40 | 44.59 | 33.48 |
| mBART-50        | 68.10 | 82.99 | 63.26 | 56.03 |
| Google Translator | 62.90 | 80.54 | 61.36 | 49.37 |
| IndicTrans      | 64.59 | 84.53 | 65.32 | 53.59 |

#### Bhagavad Gita Dataset Results
| Model           | BLEU | STS  | chrF | METEOR |
|-----------------|------|------|------|--------|
| NLLB-200        | 41.77 | 67.88 | 39.16 | 31.95 |
| M2M100          | 39.57 | 63.24 | 34.30 | 29.48 |
| mBART-50        | 43.06 | 68.96 | 36.13 | 32.38 |
| Google Translator | 48.12 | 71.34 | 46.26 | 37.78 |
| IndicTrans      | 44.84 | 68.10 | 47.07 | 35.62 |

## 🚀 Getting Started

### Prerequisites
```bash
python >= 3.8
torch >= 1.8.0
transformers >= 4.20.0
sacrebleu >= 2.0.0
sentence-transformers >= 2.2.0

## 🚀 Getting Started

### Prerequisites
```bash
python >= 3.8
torch >= 1.8.0
transformers >= 4.20.0
sacrebleu >= 2.0.0
sentence-transformers >= 2.2.0
```

### Installation
```bash
git clone https://github.com/imperialrogers/LLM-Analysis.git
cd LLM-Analysis
pip install -r requirements.txt
```

## 💻 Usage

### Loading Models
```python
from transformers import (
    MBartForConditionalGeneration, 
    M2M100ForConditionalGeneration,
    NllbTokenizer,
    AutoTokenizer
)

# Load mBART-50
mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
mbart_tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50")

# Load M2M-100
m2m_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
m2m_tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")

# Load NLLB-200
nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
nllb_tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
```

### Translation Example
```python
def translate_text(text, model, tokenizer, src_lang, tgt_lang):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
```

## 📊 Evaluation

### Running Evaluations
```python
from evaluation import calculate_metrics

# Example usage
reference = "यह एक उदाहरण वाक्य है।"
hypothesis = "This is an example sentence."

metrics = calculate_metrics(reference, hypothesis)
print(f"BLEU Score: {metrics['bleu']}")
print(f"STS Score: {metrics['sts']}")
print(f"chrF Score: {metrics['chrf']}")
print(f"METEOR Score: {metrics['meteor']}")
```

## 📁 Dataset Information

### IIT-B Hindi-English Corpus
- 18 lakh sentence pairs
- General domain text
- Source: IIT Bombay website
- 
### Bhagavad Gita Dataset
- 701 verses
- Contains Sanskrit original, romanized transliteration, Hindi and English translations
- Specialized philosophical and literary content

## 🏗️ Project Structure
```
.
├── bhagavad_gita/
│   ├── english_to_hindi/
│   │   ├── mbart.py
│   │   ├── m2m.py
│   │   ├── nllb.py
│   │   └── google_translate.py
│   └── hindi_to_english/
│       ├── mbart.py
│       ├── m2m.py
│       ├── nllb.py
│       └── google_translate.py
└── iitb_corpus/
    ├── english_to_hindi/
    │   ├── mbart.py
    │   ├── m2m.py
    │   ├── nllb.py
    │   └── google_translate.py
    └── hindi_to_english/
        ├── mbart.py
        ├── m2m.py
        ├── nllb.py
        └── google_translate.py
```

## 📝 Key Conclusions

1. **Model Performance across Domains**:
   - **mBART-50** demonstrates outstanding performance in general-purpose translations, making it a robust choice for typical language translation tasks where high accuracy is required.
   - **Google Translate** shines in translating specialized, literary, and philosophical texts (e.g., the Bhagavad Gita), where it captures nuances better than other models. This indicates its suitability for contexts where cultural or literary depth is critical.
   - **IndicTrans** performs consistently well across both general and specialized domains, highlighting its utility in versatile applications for Indian languages.

2. **Dataset Sensitivity**:
   - The analysis reveals that models like NLLB-200 and M2M100 are more sensitive to dataset variations. Their performance varies significantly between general-purpose corpora (like IIT-B) and specialized datasets (like the Bhagavad Gita), underscoring the importance of fine-tuning and domain-specific optimization for effective translation.
   
3. **Metric-Based Insights**:
   - The use of multiple evaluation metrics (BLEU, STS, chrF, METEOR) provides a more holistic view of translation quality. For instance, while BLEU gives a broad measure of accuracy, chrF captures finer details like character alignment, and STS measures semantic similarity, each highlighting different model strengths.
   - A comprehensive metric analysis helps identify strengths and limitations in each model, offering valuable insights into optimizing model selection for specific tasks.

4. **Real-World Applications and Future Scope**:
   - These findings have practical implications for multilingual applications, content localization, and cross-lingual information retrieval, particularly for the Hindi-English language pair.
   - Future work could explore fine-tuning these models on domain-specific datasets to further improve their accuracy and versatility, especially for underrepresented languages and dialects in India.

This comparison study not only aids in selecting suitable models for different types of text but also highlights the potential for further model improvements, making it a valuable resource for researchers and developers working with multilingual AI solutions.


## 👥 Contributing
Contributions are welcome! Please feel free to submit a Pull Request to the [LLM-Analysis repository](https://github.com/imperialrogers/LLM-Analysis).

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/imperialrogers/LLM-Analysis/blob/main/LICENSE) file for details.

## 📧 Contact
For questions or feedback, please [open an issue](https://github.com/imperialrogers/LLM-Analysis/issues) on the GitHub repository.
