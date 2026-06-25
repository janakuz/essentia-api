import numpy as np
import essentia.standard as es
import os
import sys
import json
import heapq
from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
EXPECTED_API_KEY = os.environ["AUDIO_API_KEY"]



async def validate_api_key(api_key: str = Depends(api_key_header)):
    if api_key == EXPECTED_API_KEY:
        return api_key
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, 
        detail="Unauthorized: Invalid or missing API Key."
    )


embedding_models = dict()
models = dict()
basic_features = dict()
class_labels = dict()

@asynccontextmanager
async def lifespan(app: FastAPI):
    embedding_model_effnet = es.TensorflowPredictEffnetDiscogs(graphFilename="model_weights/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
    embedding_model_vggish = es.TensorflowPredictVGGish(graphFilename="model_weights/audioset-vggish-3.pb", output="model/vggish/embeddings")
    embedding_models["effnet"] = embedding_model_effnet
    embedding_models["vggish"] = embedding_model_vggish

    models["approachability"] = es.TensorflowPredict2D(graphFilename="model_weights/approachability_2c-discogs-effnet-1.pb", output="model/Softmax")
    models["engagement"] = es.TensorflowPredict2D(graphFilename="model_weights/engagement_2c-discogs-effnet-1.pb", output="model/Softmax")

    models["danceability"] = es.TensorflowPredict2D(graphFilename="model_weights/danceability-discogs-effnet-1.pb", output="model/Softmax")
    models["mood_aggressive"] = es.TensorflowPredict2D(graphFilename="model_weights/mood_aggressive-discogs-effnet-1.pb", output="model/Softmax")
    models["mood_happy"] = es.TensorflowPredict2D(graphFilename="model_weights/mood_happy-discogs-effnet-1.pb", output="model/Softmax")
    models["mood_party"] = es.TensorflowPredict2D(graphFilename="model_weights/mood_party-discogs-effnet-1.pb", output="model/Softmax")
    models["mood_relaxed"] = es.TensorflowPredict2D(graphFilename="model_weights/mood_relaxed-discogs-effnet-1.pb", output="model/Softmax")
    models["mood_sad"] = es.TensorflowPredict2D(graphFilename="model_weights/mood_sad-discogs-effnet-1.pb", output="model/Softmax")

    models["moods_mtg"] = es.TensorflowPredict2D(graphFilename="model_weights/mtg_jamendo_moodtheme-discogs-effnet-1.pb")
    models["moods_mirex"] = es.TensorflowPredict2D(graphFilename="model_weights/moods_mirex-audioset-vggish-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall")

    models["instrumental"] = es.TensorflowPredict2D(graphFilename="model_weights/voice_instrumental-discogs-effnet-1.pb", output="model/Softmax")
    models["voice_gender"] = es.TensorflowPredict2D(graphFilename="model_weights/gender-discogs-effnet-1.pb", output="model/Softmax")
    models["jamendo_instruments"] = es.TensorflowPredict2D(graphFilename="model_weights/mtg_jamendo_instrument-discogs-effnet-1.pb")

    basic_features["bpm"] = es.RhythmExtractor2013(maxTempo = 250)
    basic_features["key"] = es.KeyExtractor()
    basic_features["loudness"] = es.LoudnessEBUR128()
    basic_features["dynamic"] = es.DynamicComplexity()

    with open('model_weights/mtg_jamendo_moodtheme-discogs-effnet-1.json', 'r') as f:
        metadata = json.load(f)
        class_labels["mtg"] = metadata['classes']

    class_labels["mirex"] = ["boisterous", "cheerful", "poignant", "humorous", "aggressive"]


    yield

    embedding_models.clear()
    models.clear()
    basic_features.clear()


app = FastAPI(lifespan=lifespan, dependencies=[Depends(validate_api_key)])

@app.post("/analyze")
async def analyze(file: UploadFile):
    if not file.filename.endswith(('.mp3', '.wav', '.flac', '.m4a', '.aac')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    audio_data, native_sr, num_channels, _, _, _ = es.AudioLoader(filename=temp_path)()
    audio = es.MonoMixer()(audio_data, num_channels)

    res = dict()
    
    bpm, _, _, _,  _ = basic_features["bpm"](audio)
    key, scale, _ = basic_features["key"](audio)
    _, _, integrated_loudness, _ = basic_features["loudness"](audio_data)
    dynamic_complexity, _ = basic_features["dynamic"](audio)

    res["bpm"] = bpm
    res["key"] = {"key": key, "scale":scale}
    res["loudness"] = integrated_loudness
    res["dynamic_complexity"] = dynamic_complexity

    target_sr = 16000
    if native_sr != target_sr:
        resampler = es.Resample(
            inputSampleRate=native_sr, 
            outputSampleRate=target_sr, 
            quality=4
        )
        audio_16k = resampler(audio)
    else:
        audio_16k = audio

    embeddings_effnet = embedding_models["effnet"](audio_16k)
    embeddings_vggish = embedding_models["vggish"](audio_16k)

    res["approachability"] = np.mean(models["approachability"](embeddings_effnet), axis=0)[1].item()
    res["engagement"] = np.mean(models["engagement"](embeddings_effnet), axis=0)[1].item()
    res["danceability"] = np.mean(models["danceability"](embeddings_effnet), axis=0)[0].item()
    res["mood_aggressive"] = np.mean(models["mood_aggressive"](embeddings_effnet), axis=0)[0].item()
    res["mood_happy"] = np.mean(models["mood_happy"](embeddings_effnet), axis=0)[0].item()
    res["mood_party"] = np.mean(models["mood_party"](embeddings_effnet), axis=0)[1].item()
    res["mood_relaxed"] = np.mean(models["mood_relaxed"](embeddings_effnet), axis=0)[1].item()
    res["mood_sad"] = np.mean(models["mood_sad"](embeddings_effnet), axis=0)[1].item()

    # pred_instrumental = np.mean(models["instrumental"](embeddings_effnet), axis=0)[0].item()
    # instrumental = True if pred_instrumental > 0.7 else False
    # res["pred_inst"] = pred_instrumental

    left_channel = audio_data[:, 0]
    right_channel = audio_data[:, 1]
    
    side_signal = left_channel - right_channel
    
    total_energy = np.sum(left_channel**2) + np.sum(right_channel**2)
    side_energy = np.sum(side_signal**2) * 2
    
    energy_drop_ratio = side_energy / (total_energy + 1e-6)    

    instrumental = True if energy_drop_ratio > 1.0 else False


    if not instrumental:
        voice_preds = models["voice_gender"](embeddings_effnet)

        male_probabilities = voice_preds[:, 1] 
        std_dev = np.std(male_probabilities) 
        median_male_prob = np.median(male_probabilities)


        # res["mean"] = mean_male_prob.item()
        mid_zone_frames = np.sum((male_probabilities >= 0.35) & (male_probabilities <= 0.65)) / len(male_probabilities)

        if std_dev >= 0.2 and mid_zone_frames < 0.25:
            res["voice"] = "mixed"
        else:
#            mean_voice = np.mean(voice_preds, axis=0)
            median_male_prob = np.median(male_probabilities)
            res["voice"] = "male" if median_male_prob > 0.7 else "female"
    
    res["instrumental"] = instrumental

    mtg_mood_predictions = models["moods_mtg"](embeddings_effnet)
    mean_activations = np.mean(mtg_mood_predictions, axis=0)
    predicted_moods = dict(zip(class_labels["mtg"], mean_activations))
    moods_mtg = []
    for mood, prob in predicted_moods.items():
        if prob > 0.1:
            moods_mtg.append(mood)
    
    mirex_mood_predictions = models["moods_mirex"](embeddings_vggish)
    mean_activations = np.mean(mirex_mood_predictions, axis=0)
    predicted_moods = dict(zip(class_labels["mirex"], mean_activations))
    moods_mirex = []
    max_prob = max(mean_activations)
    for mood, prob in predicted_moods.items():
        if prob >= max_prob * 0.65:
            moods_mirex.append(mood)

    res["moods"] = moods_mtg + moods_mirex

    if os.path.exists(temp_path):
        os.remove(temp_path)


    return res













    
