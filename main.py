import numpy as np
import os
import json
from pathlib import Path
import zipfile
import shutil
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import queue
import threading
import essentia.standard as es
from dotenv import load_dotenv
load_dotenv()

import time

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

dsp = dict()
embedding_models = dict()
models = dict()
class_labels = dict()

# loader = None
# mono_mixer = None
process_executor = None


def init_worker_process():
    global es
    import essentia.standard as local_es
    import essentia
    essentia.EssentiaLogger().warningActive = False

    es = local_es

    embedding_model_effnet = es.TensorflowPredictEffnetDiscogs(graphFilename="model_weights/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
    embedding_model_vggish = es.TensorflowPredictVGGish(graphFilename="model_weights/audioset-vggish-3.pb", output="model/vggish/embeddings")
    embedding_models["effnet"] = embedding_model_effnet
    embedding_models["vggish"] = embedding_model_vggish

@asynccontextmanager
async def lifespan(app: FastAPI):

    global process_executor

    ctx = multiprocessing.get_context("spawn")

    process_executor = ProcessPoolExecutor(max_workers=max(1, os.cpu_count()), 
                                           initializer=init_worker_process,
                                           mp_context=ctx,
                                           max_tasks_per_child=4)


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

    with open('model_weights/mtg_jamendo_moodtheme-discogs-effnet-1.json', 'r') as f:
        metadata = json.load(f)
        class_labels["mtg"] = metadata['classes']

    class_labels["mirex"] = ["boisterous", "cheerful", "poignant", "humorous", "aggressive"]


    yield

    process_executor.shutdown(wait=True)

app = FastAPI(lifespan=lifespan, dependencies=[Depends(validate_api_key)])


def process_dsp(temp_path):
    loader = es.AudioLoader(filename=temp_path)
    mono_mixer = es.MonoMixer()

    audio_data, native_sr, num_channels, _, _, _ = loader()
    audio = mono_mixer(audio_data, num_channels)

    res = dict()
    res["track_id"] = int(temp_path.split("/")[-1].split(".")[0])
    
    bpm = es.PercivalBpmEstimator(maxBPM=250)(audio)
    key, scale, _ = es.KeyExtractor()(audio)
    _, _, integrated_loudness, _ = es.LoudnessEBUR128()(audio_data)
    dynamic_complexity, _ = es.DynamicComplexity()(audio)

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

    left_channel = audio_data[:, 0]
    right_channel = audio_data[:, 1]
    
    side_signal = left_channel - right_channel
    
    total_energy = np.sum(left_channel**2) + np.sum(right_channel**2)
    side_energy = np.sum(side_signal**2) * 2
    
    energy_drop_ratio = side_energy / (total_energy + 1e-6)    

    instrumental = True if energy_drop_ratio > 1.0 and num_channels == 2 else False

    res["instrumental"] = instrumental

    embeddings_effnet = embedding_models["effnet"](audio_16k)
    embeddings_vggish = embedding_models["vggish"](audio_16k)


    data = {"results": res, "embeddings_effnet": embeddings_effnet, "embeddings_vggish": embeddings_vggish}

    return data


def process_ml(embeddings_effnet, embeddings_vggish, res):
    res["approachability"] = np.mean(models["approachability"](embeddings_effnet), axis=0)[1].item()
    res["engagement"] = np.mean(models["engagement"](embeddings_effnet), axis=0)[1].item()
    res["danceability"] = np.mean(models["danceability"](embeddings_effnet), axis=0)[0].item()
    res["mood_aggressive"] = np.mean(models["mood_aggressive"](embeddings_effnet), axis=0)[0].item()
    res["mood_happy"] = np.mean(models["mood_happy"](embeddings_effnet), axis=0)[0].item()
    res["mood_party"] = np.mean(models["mood_party"](embeddings_effnet), axis=0)[1].item()
    res["mood_relaxed"] = np.mean(models["mood_relaxed"](embeddings_effnet), axis=0)[1].item()
    res["mood_sad"] = np.mean(models["mood_sad"](embeddings_effnet), axis=0)[1].item()



    if not res["instrumental"]:
        voice_preds = models["voice_gender"](embeddings_effnet)

        male_probabilities = voice_preds[:, 1] 
        std_dev = np.std(male_probabilities) 
        median_male_prob = np.median(male_probabilities)


        mid_zone_frames = np.sum((male_probabilities >= 0.35) & (male_probabilities <= 0.65)) / len(male_probabilities)

        if std_dev >= 0.2 and mid_zone_frames < 0.25:
            res["voice"] = "mixed"
        else:
            median_male_prob = np.median(male_probabilities)
            res["voice"] = "male" if median_male_prob > 0.7 else "female"
    

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

    return res



@app.post("/analyze-batch")
def analyze(file: UploadFile):
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="This endpoint expects a .zip file")

    upload_dir = Path(f"/tmp/zip_{os.getpid()}_{file.filename.replace('.', '_')}")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = upload_dir / "uploaded.zip"
    
    with open(zip_path, "wb") as f:
        f.write(file.file.read())
        
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(upload_dir)

    supported_extensions = ('.mp3', '.wav', '.flac', '.m4a', '.aac', '.wma')
    audio_paths = [
        str(p) for p in upload_dir.glob("**/*") 
        if p.is_file() and p.suffix.lower() in supported_extensions
    ]
    
    if not audio_paths:
        shutil.rmtree(upload_dir)
        raise HTTPException(status_code=400, detail="No valid files in zip archive.")

    try:
        final_results = []
        ml_queue = queue.Queue()

        def ml_consumer():
            while True:
                data = ml_queue.get()
                if data is None:
                    break
            
                ml_predictions = process_ml(data["embeddings_effnet"], data["embeddings_vggish"], data["results"])
            
                final_results.append(ml_predictions)
            ml_queue.task_done()


        consumer_thread = threading.Thread(target=ml_consumer)
        consumer_thread.start()


        futures = {process_executor.submit(process_dsp, path): path for path in audio_paths}
    
        for future in as_completed(futures):
            dsp_payload = future.result()
            ml_queue.put(dsp_payload)

        ml_queue.put(None)
        consumer_thread.join()

        return {"status": "success", "results": final_results}
        
    finally:
        if upload_dir.exists():
            shutil.rmtree(upload_dir)












    
