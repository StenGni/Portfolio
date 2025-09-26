# Emotion Classification Pipeline

**Client:** [Content Intelligence Agency](https://www.linkedin.com/company/content-intelligence-agency-cia/posts/?feedView=all)
**Period:** February – April 2025  

## Overview  
In this project, we worked in a team of three to develop an **end-to-end NLP pipeline** that:  
1. **Transcribes speech** from Polish audio files (AssemblyAI & Whisper).  
2. **Translates text** into English using a custom-built machine translation model.  
3. **Classifies sentences** into one of six core emotions (happiness, sadness, anger, surprise, fear, disgust) or neutral.  

The pipeline integrates **traditional NLP methods** with modern deep learning approaches to support media analysis workflows.  

## Results  
- Annotated and cleaned a dataset of TV show transcripts for supervised training.  
- Achieved reliable transcription and evaluated **Word Error Rate (WER)**.  
- Trained and compared multiple models: Logistic Regression, Naive Bayes, LSTM, RNN, and Transformers (BERT + one alternative).  
- Best-performing model reached an **F1-score above 0.75** for multi-class classification.  
- Delivered a modular pipeline able to process new video data end-to-end.  

## Technologies  
- **Speech-to-Text:** Whisper, AssemblyAI  
- **Machine Translation:** Custom seq2seq model (HuggingFace / PyTorch)  
- **Feature Extraction:** TF-IDF, POS tagging, embeddings (Word2Vec, GloVe)  
- **Emotion Classification Models:** Logistic Regression, Naive Bayes, LSTM, RNN, Transformers (BERT-based)  
- **Explainability:** XAI methods for Transformers (Gradient × Input, LRP)  

## Data
During this project we used data available online.
- Training dataset was made from test datasets of other groups from BUas working on the same project, but with input different languages and ["Friends emotion-labeled dialogues"](https://github.com/emorynlp/emotion-detection)
- Testing dataset was made from transcription of 2 videos from Youtube([1](https://www.youtube.com/watch?v=MeX4E1kRA6w), [2](https://www.youtube.com/watch?v=uYuhEoPzT5A&t)). They were translated by AssemblyAI and emotions were labeled by CIA model. 

## License  
Educational/research use only. Contact the authors for other uses.  

## Team  
- [Monika Stangenberg](https://www.linkedin.com/in/monika-stangenberg-4b5790291/)
- [Kamil Łęga](https://www.linkedin.com/in/kamil-lega-it/)
- [Michał Bątkowski](https://www.linkedin.com/in/micha%C5%82-b%C4%85tkowski-837a8534a/)
