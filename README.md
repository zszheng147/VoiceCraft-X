# VoiceCraft-X: Unifying Multilingual, Voice-Cloning Speech Synthesis and Speech Editing

## TODO
- [x] Environment setup
- [x] Inference code for TTS and speech editing
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

## Pretrained Models
We have uploaded the pretrained models to HuggingFace. You can download them from [here](https://huggingface.co/zhisheng01/VoiceCraft-X).
```bash
cd VoiceCraft-X
git clone https://huggingface.co/zhisheng01/VoiceCraft-X pretrained_models
```

## Inference
Checkout [`speech_editing.ipynb`](./src/speech_editing.ipynb) and [`speech_synthesize.ipynb`](./src/speech_synthesize.ipynb)


## License
This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.

[![CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc/4.0/)

**Copyright (c) 2025 Zhisheng Zheng/The University of Texas at Austin**

For commercial use, please contact the authors.

## Acknowledgement
We acknowledge the following open-source projects that made this work possible:
- [AudioCraft](https://github.com/facebookresearch/audiocraft) for audio generation frameworks
- [Montreal Forced Alignment](https://montreal-forced-aligner.readthedocs.io/en/latest/) for speech alignment
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [VoiceCraft](https://github.com/jasonppy/VoiceCraft)

## Citation
If you use this work in your research, please cite:


## Disclaimer
Any organization or individual is prohibited from using any technology mentioned in this paper to generate or edit someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.