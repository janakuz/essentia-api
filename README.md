# Audio Analytics & Feature Extraction ML Microservice

A self-hosted Python REST API microservice containerized via Docker. It uses [Essentia](https://essentia.upf.edu/) to perform digital signal processing (DSP) and executes pre-trained Machine Learning models to extract detailed acoustic and mood metadata from raw audio files for the Android client.
  

## 🧠 Features

- Acoustic Feature Extraction (BPM, key and scale, and perceived loudness detection)
- Multiclass mood prediction
- Binary probabilities for emotional categories (happy, sad, aggressive, party, relaxed), danceability, as well as approachability and engagement
- Vocal analysis (instrumental vs vocal and voice gender prediction)
  

## 🛠️ Tech Stack & Infrastructure

- Python, FastAPI
- Essentia for audio analysis and music information retrieval
- Docker for containerization

## 🔗 Connected Client Application

This microservice serves endpoints explicitly consumed by the [native mobile application](https://github.com/janakuz/MusicPlayerApp).

## 📦 Deployment & Setup

The service is configured for multi-container orchestration. To pull dependencies, resolve C++ wrapped audio libraries, and spin up the microservice with a single command:

```bash
docker-compose up --build
```
