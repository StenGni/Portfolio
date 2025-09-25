# Root Segmentation & Robotic Control

**Client:** Netherlands Plant Eco-phenotyping Centre [(NPEC)](https://www.linkedin.com/company/npec-nl/posts/?feedView=all)
**Period:** November 2024 - January 2025

## Technologies  
- **Computer Vision & Segmentation:** U-Net for root masks  
- **Model Training & Inference:** PyTorch / TensorFlow  
- **Experiment Tracking:** Weights & Biases  
- **Robotics Simulation & Control:** PyBullet, Gym wrapper, PID & RL (Stable-Baselines3)  

## Overview  
In this project we developed a **pipeline** combining **computer vision, reinforcement learning, and robotics** to segment plant roots and control a liquid-handling robot for targeted inoculation.  

- Built a **CV pipeline** (ROI extraction, segmentation, RSA extraction).  
- Created a **simulation environment** (PyBullet) and Gym wrapper.  
- Implemented both **PID** or **Reinforcement Learning** controllers.  

## Results  
- **Root segmentation**: robust masks and RSA extraction.  
- **RL controller**: ~1 mm accuracy.  
- End-to-end pipeline automates inoculation with improved speed & precision.  

## License  
Educational/research use only. Contact the authors for other uses.
