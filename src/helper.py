import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import onnxruntime

from dataset import CosyVoiceTextFrontEnd
from utils import AudioTokenizer, get_diff_time_frame_and_segment
from models.voicecraftx import VoiceCraftX


def load_tokenizer(config):
    text_tokenizer = CosyVoiceTextFrontEnd(
        qwen_tokenizer_path=config.pretrained_models, 
        allowed_special="all",
        skip_zh_tn_model=config.skip_zh_tn_model
    )

    audio_tokenizer = AudioTokenizer(
        codec_type="encodec", 
        codec_ckpt_path=config.pretrained_models + "/encodec.th",
    )

    return text_tokenizer, audio_tokenizer


def text_process(text_tokenizer, prompt, target, language, task="tts", alignment_path=None):
    if task == "tts":
        if language == "chinese" or language == "japanese":
            transcript = f"<|fim_prefix|><|fim_suffix|><|fim_middle|>{prompt}{target}"
        else:
            transcript = f"<|fim_prefix|><|fim_suffix|><|fim_middle|>{prompt} {target}"
        cut_off_start, cut_off_end = None, None
    elif task == "editing":
        assert alignment_path is not None, "alignment_path is required for speech editing"
        with open(alignment_path) as align_f:
            data = [line.split(",")[:3] for line in align_f.readlines() if "words" in line]
        alignments = [{"Begin": float(begin), "End": float(end), "Label": label} for begin, end, label in data]
        result = get_diff_time_frame_and_segment(prompt, target, alignments, clean=True, language=language)
        cut_off_start, cut_off_end = result["time_frame"]

        prefix_text = result["target_prefix"].strip()
        middle_text = result["target_middle"].strip()
        suffix_text = result["target_suffix"].strip()

        transcript = f"<|fim_prefix|>{prefix_text}<|fim_suffix|>{suffix_text}<|fim_middle|>{middle_text}"

    transcript = text_tokenizer.text_normalize(transcript, split=False, text_frontend=True, lang=language)
    transcript, transcript_text_len = text_tokenizer(transcript)

    return transcript, cut_off_start, cut_off_end


def load_voicecraftx(config):
    model = VoiceCraftX(config.model)

    ckpt = torch.load(config.voicecraftx_path, map_location="cpu")["state_dict"]
    for k in list(ckpt.keys()):
        if k.startswith("model."):
            ckpt[k[6:]] = ckpt.pop(k)
    model.load_state_dict(ckpt, strict=False)
    return model.eval()


def load_speaker_model(config):
    campplus_model = config.pretrained_models + "/speech_campplus.onnx"
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    campplus_session = onnxruntime.InferenceSession(
        campplus_model, sess_options=option, providers=["CPUExecutionProvider"]
    )
    return campplus_session


def extract_speaker_embedding(campplus_session, speech):
    feat = kaldi.fbank(
        speech,
        num_mel_bins=80,
        dither=0,
        sample_frequency=16000
    )
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = campplus_session.run(None, {
        campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()
    })[0].flatten()
    return torch.tensor(embedding)


def generate(
    prompt_audio, prompt_text, target_text, 
    config, model=None, speaker_model=None, 
    text_tokenizer=None, audio_tokenizer=None, 
    task="tts", alignment_path=None, 
    language="english", n_samples=5, device="cuda"
):
    generation_config = {
        "max_length": config.MAX_LENGTH * config.CODEC_FRAME_RATE,
        "min_p": 0.0, # 0.01-0.2 range. if 0.0 then use top_k/top_p sampling.
        "top_k": 20, "top_p": 1.0, 
        "temperature": 1.0
    }
    transcript, cut_off_start, cut_off_end = text_process(
        text_tokenizer=text_tokenizer, 
        prompt=prompt_text, target=target_text, 
        language=language, task=task, alignment_path=alignment_path
    )
    transcript = transcript.to(device)
    
    audio, orig_sr = torchaudio.load(prompt_audio)
    if orig_sr != config.SAMPLE_RATE:
        audio = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=config.SAMPLE_RATE)(audio)

    spk_embedding = extract_speaker_embedding(speaker_model, audio).to(device)
    audio_input_ids = audio_tokenizer(audio.unsqueeze(0).to(device))

    cushion = torch.tensor(config.model.silence_tokens, device=device).view(1, 4, 1).expand(1, 4, 3)
    mask = model.speech_mask.unsqueeze(0)
    

    audio_prefix = audio_input_ids[..., :round(cut_off_start*config.CODEC_FRAME_RATE)] \
        if cut_off_start is not None else cushion
    audio_suffix = audio_input_ids[..., round(cut_off_end*config.CODEC_FRAME_RATE):] \
        if cut_off_end is not None else cushion
    audio_middle = torch.cat([audio_input_ids, cushion], dim=-1) \
        if (cut_off_start is None and cut_off_end is None) else cushion
    
    audio_input_ids = torch.cat([audio_prefix, mask, audio_suffix, mask, audio_middle], dim=-1)

    with torch.no_grad():
        outputs = model.generate(
            n_samples=n_samples,
            text=transcript,
            prompt_speech_token=audio_input_ids,
            speaker_emb=spk_embedding,
            generation_config=generation_config
        )
    outputs = [output.unsqueeze(0) for output in outputs]
    if task == "editing":
        outputs = [
            torch.cat([audio_prefix, cushion, output, cushion, audio_suffix], dim=-1) 
            for output in outputs
        ]

    return outputs