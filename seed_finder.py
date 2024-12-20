import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import numpy as np
from pathlib import Path
from typing import List, Tuple
import torch.nn.functional as F
import requests
import json
import tempfile
import os
from tqdm import tqdm
import time
from datetime import timedelta

class AudioEmbedder:
    def __init__(self):
        # Initialize wav2vec2 model and processor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use the fine-tuned model that includes all necessary files
        model_name = "facebook/wav2vec2-large-robust-ft-swbd-300h"
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def get_embedding(self, audio_path: str) -> np.ndarray:
        """
        Extract embedding from an audio file using wav2vec2
        """
        # Load and resample audio to 16kHz (required for wav2vec2)
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Process audio
        with torch.no_grad():
            inputs = self.processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            # Use mean pooling over time dimension to get fixed-size embedding
            embedding = torch.mean(outputs.last_hidden_state, dim=1)
            
        return embedding.cpu().numpy().squeeze()

def generate_audio(text: str, seed: int, api_url: str = "http://82.130.14.108:5000/predict") -> bytes:
    """Generate audio for given text and seed"""
    payload = {
        "text": text,
        "reference": "https://firebasestorage.googleapis.com/v0/b/audiobookgen.appspot.com/o/testoo%2FElevenLabs_2024-12-17T19_46_35_Female%20Narrator%201_gen_s50_sb75_se0_b_m2.wav?alt=media&token=29306783-d151-4556-9c41-d9481207cad4",
        "weights": "l26AVwIi7QuBeQguFlXa_ep19.pth",
        "diffusion_steps": 80,
        "seed": seed
    }
    
    response = requests.post(api_url, json=payload)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Generation failed with status code: {response.status_code}")

def evaluate_seed(seed: int, texts: List[str], reference_paths: List[str], embedder: AudioEmbedder, output_dir: Path) -> Tuple[float, float]:
    """Generate and evaluate audio for a single seed"""
    scores = []
    
    # Check if both files for this seed already exist
    file1 = output_dir / f"seed_{seed:04d}_text_1.wav"
    file2 = output_dir / f"seed_{seed:04d}_text_2.wav"
    
    if file1.exists() and file2.exists():
        # If both files exist, just evaluate them
        for idx, ref_path in enumerate(reference_paths):
            gen_path = output_dir / f"seed_{seed:04d}_text_{idx+1}.wav"
            ref_embedding = embedder.get_embedding(ref_path)
            gen_embedding = embedder.get_embedding(str(gen_path))
            
            distance = 1 - F.cosine_similarity(
                torch.tensor(ref_embedding).unsqueeze(0),
                torch.tensor(gen_embedding).unsqueeze(0)
            ).item()
            scores.append(distance)
            
        return scores[0], scores[1]
    
    # If files don't exist, generate them
    for idx, (text, ref_path) in enumerate(zip(texts, reference_paths)):
        output_path = output_dir / f"seed_{seed:04d}_text_{idx+1}.wav"
        
        if not output_path.exists():
            # Generate audio for this seed
            audio_data = generate_audio(text, seed)
            # Save the audio file
            output_path.write_bytes(audio_data)
        
        # Get embeddings and calculate distance
        ref_embedding = embedder.get_embedding(ref_path)
        gen_embedding = embedder.get_embedding(str(output_path))
        
        # Calculate cosine distance
        distance = 1 - F.cosine_similarity(
            torch.tensor(ref_embedding).unsqueeze(0),
            torch.tensor(gen_embedding).unsqueeze(0)
        ).item()
        
        scores.append(distance)
    
    return scores[0], scores[1]

def find_best_seeds(texts: List[str], reference_paths: List[str], start_seed: int = 0, end_seed: int = 999):
    """Find the best seeds for both texts"""
    # Create output directory
    output_dir = Path("generated_audio")
    output_dir.mkdir(exist_ok=True)
    
    # Save configuration info
    config = {
        "start_seed": start_seed,
        "end_seed": end_seed,
        "texts": texts,
        "reference_paths": [str(p) for p in reference_paths],
        "timestamp": time.strftime("%Y%m%d_%H%M%S")
    }
    with open(output_dir / "generation_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    embedder = AudioEmbedder()
    results = []
    
    total_seeds = end_seed - start_seed + 1
    start_time = time.time()
    last_generation_time = None
    
    print(f"\nProcessing {total_seeds} seeds, 2 generations each...")
    print(f"Saving audio files to: {output_dir.absolute()}")
    
    for seed in range(start_seed, end_seed + 1):
        iteration_start = time.time()
        try:
            score1, score2 = evaluate_seed(seed, texts, reference_paths, embedder, output_dir)
            avg_score = (score1 + score2) / 2
            results.append((seed, score1, score2, avg_score))
            
            # Calculate timing information
            iteration_time = time.time() - iteration_start
            last_generation_time = iteration_time
            seeds_remaining = end_seed - seed
            estimated_time_remaining = seeds_remaining * iteration_time
            
            # Format progress message
            progress = f"Seed {seed}/{end_seed}"
            timing = f"Last generation: {iteration_time:.1f}s"
            eta = f"ETA: {str(timedelta(seconds=int(estimated_time_remaining)))}"
            
            print(f"\r{progress} | {timing} | {eta}", end="")
            
        except Exception as e:
            print(f"\nError processing seed {seed}: {str(e)}")
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n\nProcessing Complete!")
    print(f"Total time: {str(timedelta(seconds=int(total_time)))}")
    print(f"Average time per seed: {(total_time/total_seeds):.1f}s")
    
    # Sort by average score
    results.sort(key=lambda x: x[3])
    
    # Print top 10 results
    print("\nTop 10 Seeds:")
    print("Seed\tScore 1\tScore 2\tAverage")
    print("-" * 40)
    for seed, s1, s2, avg in results[:10]:
        print(f"{seed}\t{s1:.4f}\t{s2:.4f}\t{avg:.4f}")
    
    # Save results to file
    results_file = output_dir / "results.json"
    results_data = {
        "all_results": results,
        "total_time": total_time,
        "total_seeds": total_seeds,
        "average_time_per_seed": total_time / total_seeds
    }
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)

if __name__ == "__main__":
    texts = [
        "The city stretched below like an ocean of glass and whispers, its pulse hidden beneath the hum of traffic and flickering streetlights. On the rooftop, Eleanor tightened her scarf against the wind, the scent of rain and rust clinging to the air.",
        "The rain poured in sheets, drenching the cobblestones beneath his feet as he stood, fists clenched, in the empty square. 'Say it!' he shouted, his voice cracking against the storm's roar. Across from him, Isabella froze, her silhouette trembling under the dim light of a flickering lamppost. Her lips parted as if to speak, but no words came. 'Say it, and I'll walk away,' he pleaded, the thunder swallowing his last words."
    ]
    
    reference_paths = ["reference_output_1.wav", "reference_output_2.wav"]
    
    find_best_seeds(texts, reference_paths, end_seed=999)