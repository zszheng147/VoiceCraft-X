# VoiceCraft-X: Unifying Multilingual, Voice-Cloning Speech Synthesis and Speech Editing

## TODO
- [x] Environment setup
- [x] Inference code for TTS and speech editing
- [ ] Training/finetuning guidance
- [ ] Model weights
- [ ] HuggingFace Spaces demo
- [ ] Colab notebooks
- [ ] Command line
- [ ] Improve efficiency


## Installation
```bash
conda create -n voicecraftx python=3.10
conda activate voicecraftx

# montreal-forced-aligner is for speech editing, if you find it difficult to install, you can ignore it.
conda install -c conda-forge montreal-forced-aligner==3.2.1 
pip install -r requirements.txt
```

## Inference
Checkout [`speech_editing.ipynb`](./src/inference/speech_editing.ipynb) and [`speech_synthesize.ipynb`](./src/inference/speech_synthesize.ipynb)


## License

## Acknowledgement

## Citation


## Disclaimer
Any organization or individual is prohibited from using any technology mentioned in this paper to generate or edit someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.