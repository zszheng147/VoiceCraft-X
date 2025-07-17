# Copyright (c) 2025 Zhisheng Zheng/The University of Texas at Austin
#
# This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
# You may obtain a copy of the License at: http://creativecommons.org/licenses/by-nc/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn


class AudioTokenizer(nn.Module):
    def __init__(self, codec_type, codec_ckpt_path=None) -> None:
        super().__init__()
        self.codec_type = codec_type
        self.codec_ckpt_path = codec_ckpt_path

        from audiocraft.solvers import CompressionSolver
        self.model = CompressionSolver.model_from_checkpoint(self.codec_ckpt_path, "cpu")

        self.model.eval()

    def forward(self, x, *args, **kwargs):
        return self.encode(x, *args, **kwargs)

    @property
    def device(self):
        return next(self.parameters()).device
    
    def encode(self, wav: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.model.encode(wav)[0]

    def decode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.model.decode(frames)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_tokenizer = AudioTokenizer(
        codec_type="encodec", 
        codec_ckpt_path="/home/ubuntu/VoiceCraft-X/pretrained_models/encodec.th",
    ).to(device)
    
    audio = torch.zeros(1, 1, 3200).to(device)
    print(audio_tokenizer(audio))