# Emotion Detection Pipeline & Cloud Application

**Client:** [Content Intelligence Agency](https://www.linkedin.com/company/content-intelligence-agency-cia/posts/?feedView=all)
**Period:** April – June 2025 

📌 **Note:** For a visual overview of the project, please check the file **Presentation Poster** in this repository.  

⚠️ **Code Notice:** The actual implementation (pipelines, APIs, deployments) is **not included in this portfolio repository**, since it contains API keys and other sensitive configuration files.

---

## Overview  
This project focused on productionising an [**emotion classification pipeline**](https://github.com/StenGni/Portfolio/tree/main/Emotion%20Classification) and deploying it as a **cloud-based application**.  
The system processes **video/audio input**, transcribes speech, translates into English, and classifies each sentence into one of six core emotions (anger, disgust, fear, happiness, sadness, surprise) or custom emotions.  

- Modular **Python package** built from PoC notebooks and scripts.  
- **Web interface, CLI, and API** for interaction.  
- **Azure Machine Learning** for training, deployment, monitoring, and CI/CD.  
- Secure, multi-user, scalable design following **MLOps best practices**.  

## Results  
- Automated pipeline: transcription → translation → emotion tagging.  
- Deployed on **Azure** as a containerised application.  
- Accessible via **web app, REST API, and CLI**.  
- Implemented monitoring, logging, and retraining strategies.  

## Technologies  
- **NLP & Classification:** Transformers (HuggingFace), traditional ML baselines  
- **Model Training & Deployment:** PyTorch, Azure Machine Learning  
- **Experiment Tracking & CI/CD:** MLflow, GitHub Actions  
- **Containerisation:** Docker  
- **API & Frontend:** FastAPI, optional Streamlit/Gradio interface  

## License  
Educational/research use only. Contact the authors for other uses.  

## Team  
- [Monika Stangenberg](https://www.linkedin.com/in/monika-stangenberg-4b5790291/)
- [Kamil Łęga](https://www.linkedin.com/in/kamil-lega-it/)
- [Sasha Stacie](https://www.linkedin.com/in/sasha-stacie/)
- [Deuza Varela](https://www.linkedin.com/in/deuza-varela/)
- [Celine Wu](https://www.linkedin.com/in/celine-wu-8494672a8/)