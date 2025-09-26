# Model Card: Emotion Classification Transformer (RoBERTa)

## Model Overview

This project presents an emotion classification model based on the RoBERTa transformer architecture. The model was trained on cleaned and labelled text data to predict emotional categories in a 7-class setup (6 emotions + neutral). The goal was not just to create a functional classifier, but to explore broader responsible AI goals, such as handling noisy annotations, ensuring fair performance across imbalanced classes, and aligning model design with sustainable AI practices.

Throughout development, the model's behaviour was closely monitored via structured evaluation and error analysis, revealing key insights into its strengths, confusion tendencies, and performance sensitivity to label quality. This analysis has directly shaped model iterations and highlighted critical issues such as overconfidence in misclassifications, especially in short, ambiguous text.

---

## Architecture

This model represents iteration v12 of the project’s development cycle. It reflects multiple rounds of tuning, evaluation, and explainability diagnostics. The final training run, logs, and configuration are documented in the ([training notebook](https://github.com/BredaUniversityADSAI/2024-25c-fai2-adsai-group-group22_y2c/blob/main/task%205/Transformer/task5_transformerV12.ipynb)). The resulting model checkpoint is publicly accessible for loading and inference via OneDrive checkpoint ([link](https://edubuas-my.sharepoint.com/:f:/g/personal/232079_buas_nl/EhXsvXuQLadFsAZhDZQUvE0B64do7DHxuo62gwG4wgg7og?e=o6fhfl)).

The model utilizes the `roberta-base` architecture from Hugging Face, fine-tuned for multi-class emotion classification.

The technical architecture includes:

- **Transformer Backbone:** RoBERTa with pre-trained language modeling weights
- **Classification Head:** A softmax layer applied over the pooled output
- **Loss Function:** CrossEntropyLoss with class weights to address class imbalance
- **Optimization:**
  - Mixed-precision training (FP16)
  - Stratified oversampling of underrepresented classes
  - Optuna hyperparameter tuning across 5 folds
  - Early stopping (patience = 3 epochs)

**Final Hyperparameters:**

- Learning rate: `1.47e-5`
- Weight decay: `1.27e-6`
- Batch size: `64`
- Epochs: up to `10`

The training leveraged a custom Trainer implementation to compute class-weighted loss and evaluate performance on two external test sets in parallel.

---

## Purpose

The model was designed to classify emotions from short text passages, such as those found in user comments, transcripts, and conversational exchanges. The main goals of the project include:

- **Supporting Media Analytics:** By detecting emotional tone in public and consumer-generated content.
- **Evaluating AI Fairness and Sustainability:** Through analysis of imbalanced class performance and training efficiency.
- **Exploring Explainability Techniques:** Using SHAP and attention visualizations to interpret predictions.

The educational purpose was to build and document a robust, transparent, and responsibly developed emotion classifier under resource and time constraints.

---

## Development Context

This project was developed over a 3-week period using a local GPU machine supporting FP16 precision. Several limitations shaped the model’s design:

- **Label Noise:** Roughly 33% of the test set was mislabeled (based on later correction), heavily influencing initial evaluation metrics.
- **Data Imbalance:** Classes such as "disgust" and "surprise" were underrepresented.
- **Ambiguous Inputs:** Many inputs were short or context-free, limiting emotion interpretability.

To address these challenges, several techniques were employed:

- Class rebalancing through oversampling
- Stratified train/val/test splits
- Partial reallocation of test samples to validation for stability
- Evaluation on both original (`test.csv`) and corrected (`test_ad.csv`) test sets

---

## Intended Use

This model is designed for research, prototyping, and exploratory deployments in NLP pipelines, particularly for content moderation, sentiment analysis, and media research.

### Intended Applications:

- Emotion analysis in chat or interview transcripts
- Social media monitoring tools
- Tagging systems in video subtitle pipelines

### Out of Scope:

- Mental health diagnostics or clinical applications
- Multimodal emotion recognition
- Use in non-English or low-resource languages without adaptation

Due to its strong reliance on lexical emotion cues, the model should be used with caution in settings requiring nuanced or implicit sentiment interpretation.

---

## Dataset Details

One of the core considerations of this project is the intended application to Polish-language text. However, due to limited access to labeled Polish emotion datasets and compatibility issues with SpaCy for preprocessing Polish text, the training was conducted on English-translated versions of originally Polish utterances. This pipeline transcription, translation, then inference forms the foundation of the model’s envisioned deployment. In practice, spoken Polish dialogue is first transcribed, then translated into English, and finally passed through this model for emotion classification.

This constraint influenced many design decisions and also emphasizes the multilingual, cross-lingual ambition of the pipeline. The translation and preprocessing pipeline is documented in the project repository and should be referenced when interpreting the final model’s output and generalizability. ([see pipeline code](https://github.com/BredaUniversityADSAI/2024-25c-fai2-adsai-group-group22_y2c/blob/main/task%2011/Pipeline_v4.ipynb))

The model was trained on the internal `Week7_datasetV2`, consisting of English sentences labeled across seven emotion categories. ([access dataset CSV](https://github.com/BredaUniversityADSAI/2024-25c-fai2-adsai-group-group22_y2c/blob/main/task%205/Transformer/Week7_datasetV2.csv))

**Key Details:**

- **Label Format:** String → Integer encoded (LabelEncoder)
- **Preprocessing Steps:**
  - Removed nulls and duplicate entries
  - Applied stratified sampling for fair class representation
  - Performed resampling to address class imbalance
  - Added 15% of corrected test set (`test_ad.csv`) to validation split

**Final Data Split:**

- Train: 72%
- Validation: 8% + 15% (reallocated from test)
- Test: 20% (two sets: original and corrected)

## Language and Cultural Representation

Although the model is intended to classify Polish-language utterances (via English translation), the training data itself was drawn from a much broader linguistic base. The final dataset used for training (Week7_datasetV2) was compiled from multiple group contributions and includes content originally in French, German, Dutch, Bulgarian, Romanian, Russian, Hungarian, Polish, Arabic, and English. All source utterances were transcribed from real-world spoken media, including reality TV and YouTube videos. These were later translated into English by their respective groups for use in their own test sets, we repurposed them as training data to enrich diversity and generalizability.

This translation-centric preprocessing approach introduces both opportunities and limitations. While it broadens cultural context and helps avoid overfitting to a single linguistic style, it may also strip away some language-specific emotional subtleties. In practice, this means the model was trained on a diverse but monolingual (English) dataset, with cross-lingual emotional intent preserved only via translation.

The test data, however, was derived exclusively from translated Polish speech, reflecting our actual target use case. This makes it possible to analyze cross-lingual generalization performance by comparing training on multilingual-translated data and evaluation on a single-language-translated test set.

---

## Performance Metrics and Evaluation

Model performance was rigorously evaluated using accuracy, precision, recall, and weighted F1. The corrected test set (`test_ad.csv`) provided a more accurate benchmark of model behavior.

| Metric    | test.csv | test\_ad.csv |
| --------- | -------- | ------------ |
| Accuracy  | 0.544    | 0.533        |
| Precision | 0.65     | 0.65         |
| Recall    | 0.544    | 0.533        |
| F1 Score  | 0.460    | **0.538**    |

This improved F1 on `test_ad.csv` is notable, given that 137 of the corrected labels had been correctly predicted by the model, suggesting that the original labels underestimated model accuracy.

Confusion matrix analysis ([see full error analysis](https://github.com/BredaUniversityADSAI/2024-25c-fai2-adsai-group-group22_y2c/blob/main/task%208/Task%208%20-%20Error%20Analysis.pdf)) showed persistent issues:

- Surprise → Sadness (71 misclassifications)
- Disgust → Anger (40 misclassifications)
- Happiness → Neutral (13 misclassifications)

These errors reflect semantic overlaps and short, ambiguous inputs (1–6 tokens), which lack the context RoBERTa needs to disambiguate emotional tone.

Confidence calibration was also problematic: many misclassifications occurred at >0.9 confidence, highlighting the need for calibration tools or uncertainty-aware thresholds.

---

## Explainability and Transparency

A detailed [XAI analysis](https://github.com/BredaUniversityADSAI/2024-25c-fai2-adsai-group-group22_y2c/blob/main/task%209/task9.ipynb) was conducted to uncover how the model makes its predictions and where its explanations may falter.

### Qualitative Review of Predictions

A set of 21 hand-selected test samples (3 per emotion class) revealed the following trends:

- **Happiness and Neutral** classes were generally well-understood by the model, with correct classifications often focusing on polite expressions or descriptive positivity (e.g., "Thank you kindly").
- **Disgust and Surprise** were misclassified consistently. Disgust was frequently confused with anger due to shared lexical negativity. Surprise was misread as neutral, often because the model failed to understand interrogative structures (e.g., "Where did you come from?").
- **Fear and Sadness** had partial misclassifications, often tied to subtle emotional language or lack of overt sentiment markers.

These patterns indicate that the model relies heavily on explicit emotion cues and struggles with abstract or context-dependent tones.

### Token-Level Feature Importance

Through SHAP and attention-weight methods:

- Tokens like “thank”, “you”, “interview”, and “beautiful” had strong positive contributions for **happiness**.
- Negative emotions like **anger** and **disgust** shared high-activation words (e.g., "jerk", "aggression"), indicating potential semantic overlap in internal representations.
- For **surprise**, words like “Does” or punctuation (“?”) often had weak or misleading contributions.

This analysis suggests the model overemphasizes certain emotionally charged words, sometimes ignoring broader sentence semantics or structure.

### Confidence and Misclassification Trends

Examples showed many misclassified predictions had very high confidence scores (e.g., > 0.9), particularly for short inputs. This reflects a confidence calibration issue: the model often expresses strong certainty even when incorrect, undermining interpretability and reliability.

A histogram illustrating confidence distribution for correct vs incorrect predictions should be included here to demonstrate overconfidence visually. *(Insert Figure: Confidence Calibration Histogram)*

### Input Perturbation: Robustness Check

Input reduction experiments showed that the model often latches onto a few emotionally salient tokens. When those are removed:

- Performance and confidence may dip, but often recover as long as other lexical signals remain.
- For short or sarcastic inputs, robustness was low. Removing a key word like “great” or “pity” could change the outcome entirely.

This confirms that while the model can adapt to some perturbation, it lacks broader contextual reasoning and distributes attention unevenly across inputs.

### Implications

These explainability results reinforce the need for careful calibration and label quality. They also justify introducing prediction thresholds, human-in-the-loop validation, and possibly ensemble modeling if deployed in sensitive domains.

Suggested visualizations to be added:

- SHAP summary plot of token impact across classes
- Examples of token attention heatmaps for misclassified sentences
- Histogram comparing confidence of correct vs incorrect predictions

---

## Recommendations for Use

To ensure optimal performance and ethical deployment, users should consider the following:

- Use the model with a prediction confidence threshold, especially in decision-critical pipelines
- Avoid relying on it for very short texts unless context is provided externally
- Conduct human review for edge cases in datasets with label noise

### For Media Stakeholders:

- Integrate the model into media analysis pipelines with visual dashboards
- Use insights from error analysis to build correction layers or confidence-triggered review flags

---

## Sustainability Considerations

Sustainability is a critical consideration in the development and deployment of modern AI systems. Even relatively small-scale models can accumulate a non-trivial energy footprint during experimentation and repeated use. Our RoBERTa-based emotion classifier was developed with an awareness of the environmental impact of transformer architectures, and several concrete steps were taken to reduce its carbon footprint throughout the pipeline.

### Training Energy Use

The model was fine-tuned on a server equipped with an NVIDIA L40S GPU (TDP ≈ 300W) over approximately 25 minutes.

**Estimated energy consumption:**
**Energy = Runtime × Power Draw / 1000 = 0.42 h × 300 W / 1000 = 0.126 kWh**

- Charging a smartphone \~42 times
- Running a microwave for 30 minutes

Considering that multiple versions of the model were trained during the experimentation phase, the cumulative energy use likely exceeded 3–5 kWh. While relatively low in absolute terms, it highlights the value of training efficiency even at the educational level.

### Inference Efficiency

Inference was conducted locally and in batches. Each inference (prediction) takes approximately 0.2 seconds, which at 170W implies:

**Energy per inference ≈ (170W × 0.2s) / 3600 ≈ 0.009 Wh**

If the model were deployed in a small-scale application with 10,000 monthly inferences, the projected monthly inference energy use would be:

**10,000 × 0.009 Wh = 90 Wh = 0.09 kWh/month**

While small, this footprint would scale with usage. In production, inference costs may exceed training costs due to repeated usage.

### Sustainable Practices in Development

Several practices were integrated into the development process to align with green AI principles:

- **Early Stopping:** Prevented wasteful training beyond performance plateaus
- **Stratified Sampling:** Reduced the need for aggressive data augmentation or rebalancing
- **Reuse of Pretrained Models:** Leveraged Hugging Face's `roberta-base` to avoid training from scratch
- **Batch Inference:** Reduced GPU idle power and overhead during evaluation
- **Lightweight Local Deployment:** Avoided dependence on high-energy cloud infrastructure

These choices reflect a broader commitment to ethical and energy-conscious AI design.

### Ethical and Responsible AI Alignment

In addition to energy usage, the model aligns with ethical AI principles:

- It avoids deployment in high-risk or medical settings
- Human review was used to validate test data and improve label quality
- Confidence thresholds can be tuned to reduce harm from overconfident misclassifications

Moreover, [our XAI analysis](https://github.com/BredaUniversityADSAI/2024-25c-fai2-adsai-group-group22_y2c/blob/main/task%209/task9.ipynb) revealed that the model occasionally overemphasized punctuation or emotionally charged words without context. This insight encourages transparency around model limitations and reinforces the importance of explainable and interpretable NLP systems.

### Opportunities for Greener NLP

To further reduce the environmental impact of this model in future versions, we recommend:

- Deploying a distilled version such as `DistilRoBERTa`
- Quantizing the model to 8-bit or 4-bit for inference
- Tracking energy use via tools like CodeCarbon
- Considering CPU-based inference for lightweight, low-volume deployments

By embedding these considerations into our design and evaluation pipeline, we aim to contribute to a culture of responsible, sustainable AI development — where performance, interpretability, and environmental impact are equally prioritized.

---

## Licensing and Sharing

- **License:** Educational use only
- **Repository:** [GitHub Repo](https://github.com/BredaUniversityADSAI/2024-25c-fai2-adsai-group-group22_y2c)
- **Error Analysis + XAI:** Linked in GitHub repo and referenced throughout this card

---

## Acknowledgements

This model was developed by **Group 22** as part of the NLP curriculum at **Breda University of Applied Sciences**, under the supervision of Jason Harty.

The structure and formatting of this model card was inspired by Meta’s LLaMA model documentation. Evaluation and writing were supported using OpenAI's ChatGPT for idea generation and planning, together with supporting contents such as conducted error analysis, and XAI sheet developed by the students. All content was authored and reviewed by the students.

