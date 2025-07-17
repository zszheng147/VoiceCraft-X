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

import re
import unicodedata


def get_all_unicode_punctuation():
    """
    Get all Unicode punctuation characters.
    """
    punctuation_chars = set()
    for codepoint in range(0x110000):  # Unicode 范围到 U+10FFFF
        char = chr(codepoint)
        # 判断 Unicode 分类是否以 'P' 开头（如 Po、Pd 等）
        if unicodedata.category(char).startswith('P'):
            punctuation_chars.add(char)
    return ''.join(sorted(punctuation_chars))

punctuation_all = get_all_unicode_punctuation()

# Language categories for tokenization
LANGUAGES_WORD_BASED = {"english", "spanish", "dutch", "french", "german", "italian", "portuguese", "polish", "korean"}
LANGUAGES_CHAR_BASED = {"chinese", "japanese"}

def remove_punctuation(s):
    """Removes all punctuation from a string."""
    return re.sub(f"[{re.escape(punctuation_all)}]", "", s)

def build_mapping(original):
    """
    Builds a mapping from the cleaned string to the original character indices.
    Returns:
       mapping: A list where mapping[i] is the index of the i-th character of the cleaned string in the original string.
       cleaned: The string after removing punctuation (preserving spaces and other non-punctuation characters).
    """
    mapping = []
    cleaned_chars = []
    for idx, ch in enumerate(original):
        if ch not in punctuation_all:
            mapping.append(idx)
            cleaned_chars.append(ch)
    return mapping, "".join(cleaned_chars)

def build_mapping_tokens(original):
    """
    For English text: after removing punctuation (preserving spaces),
    uses regex matching to find the position of each word in the cleaned text,
    and then maps it back to the boundaries in the original text.
    
    Returns:
       token_spans: A list where each element is (start_index, end_index) (original text indices, end is non-inclusive).
       target_clean: The text after removing punctuation (spaces preserved).
    """
    mapping_chars, target_clean = build_mapping(original)
    token_spans = []
    for match in re.finditer(r'\S+', target_clean):
        start_clean, end_clean = match.span()
        # mapping_chars gives the position of each cleaned character in the original text
        orig_start = mapping_chars[start_clean]
        orig_end = mapping_chars[end_clean - 1] + 1
        token_spans.append((orig_start, orig_end))
    return token_spans, target_clean

def get_diff_time_frame_and_segment(prompt_text, target_text, alignments, clean=True, language="english"):
    """
    Finds the common prefix and common suffix between prompt_text and target_text,
    and returns:
      - time_frame: A tuple (end time of the last word/char of the common prefix,
                               start time of the first word/char of the common suffix)
                     Returns None on the respective side if prefix or suffix is missing.
      - target_text segmented into three parts: prefix, middle, suffix

    Parameters:
      - clean: Whether to remove punctuation first (alignment is based on cleaned text,
               result returns segmentation of the original text).
      - language: "chinese", "english", "japanese", "korean", "spanish", "dutch", 
                  "french", "german", "italian", "portuguese", "polish"
                  Determines tokenization strategy (character-based or word-based).
    """
    if language not in LANGUAGES_WORD_BASED and language not in LANGUAGES_CHAR_BASED:
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Supported word-based: {LANGUAGES_WORD_BASED}, "
            f"Supported char-based: {LANGUAGES_CHAR_BASED}"
        )

    if clean:
        prompt_clean = remove_punctuation(prompt_text) # prompt_text is always cleaned this way if clean=True
        if language in LANGUAGES_WORD_BASED:
            # Word-based: build token spans for target_text after cleaning
            token_spans, target_clean = build_mapping_tokens(target_text)
        elif language in LANGUAGES_CHAR_BASED:
            # Character-based: build char mapping for target_text after cleaning
            mapping_target, target_clean = build_mapping(target_text)
        # No else needed due to the check at the beginning of the function
    else: # Not cleaning
        prompt_clean = prompt_text
        target_clean = target_text
        if language in LANGUAGES_WORD_BASED:
            token_spans = []
            for match in re.finditer(r'\S+', target_text): # Use target_text (which is target_clean here)
                token_spans.append(match.span())
        elif language in LANGUAGES_CHAR_BASED:
            mapping_target = list(range(len(target_text))) # Use target_text (which is target_clean here)
        # No else needed

    # Tokenize prompt and target based on language type
    if language in LANGUAGES_WORD_BASED:
        prompt_tokens = prompt_clean.split()
        target_tokens = target_clean.split()
    elif language in LANGUAGES_CHAR_BASED:
        prompt_tokens = list(prompt_clean)
        target_tokens = list(target_clean)
    # No else needed
    
    # Calculate common prefix length (token by token comparison)
    prefix_len = 0
    for token_p, token_t in zip(prompt_tokens, target_tokens):
        if token_p == token_t:
            prefix_len += 1
        else:
            break

    # Calculate common suffix length (compare from back to front)
    suffix_len = 0
    for token_p, token_t in zip(prompt_tokens[::-1], target_tokens[::-1]):
        if token_p == token_t:
            suffix_len += 1
        else:
            break

    # Avoid prefix/suffix overlap
    if prefix_len + suffix_len > len(prompt_tokens):
        suffix_len = len(prompt_tokens) - prefix_len

    # Calculate time boundaries based on alignment data
    # If prefix exists, take the end time of the last token of the prefix in prompt_text; otherwise None
    if prefix_len > 0:
        diff_start_time = alignments[prefix_len - 1]["End"]
    else:
        diff_start_time = None
    # If suffix exists, take the start time of the first token of the suffix in prompt_text; otherwise None
    if suffix_len > 0:
        diff_end_time = alignments[len(prompt_tokens) - suffix_len]["Begin"]
    else:
        diff_end_time = None

    # Map the split points of the cleaned target_text back to positions in the original text
    if language in LANGUAGES_WORD_BASED:
        if prefix_len > 0:
            prefix_end_original = token_spans[prefix_len - 1][1]
        else:
            prefix_end_original = 0
        if suffix_len > 0:
            # Ensure len(token_spans) is checked if it can be less than len(target_tokens)
            # Assuming token_spans corresponds to target_tokens derived from target_clean
            suffix_start_original = token_spans[len(target_tokens) - suffix_len][0]
        else:
            suffix_start_original = len(target_text)
    elif language in LANGUAGES_CHAR_BASED:
        if prefix_len > 0:
            prefix_end_original = mapping_target[prefix_len - 1] + 1
        else:
            prefix_end_original = 0
        if suffix_len > 0:
            suffix_start_original = mapping_target[len(target_clean) - suffix_len]
        else:
            suffix_start_original = len(target_text)
    # No else needed

    # Store the initial boundaries based on content match
    p_content_end = prefix_end_original
    s_content_start = suffix_start_original

    # Adjust boundaries to include surrounding punctuation/whitespace
    if suffix_len == 0 and all(ch in punctuation_all or ch.isspace() for ch in target_text[p_content_end:]):
        # If no suffix and the rest of the string after prefix content is just punctuation/space,
        # then the entire target_text becomes the prefix.
        prefix_end_original = len(target_text)
        suffix_start_original = len(target_text)
    else:
        # Adjust prefix_end_original to expand rightwards for punctuation/space
        # Stop if we hit the start of the original suffix *content* (s_content_start)
        current_pe = p_content_end
        while current_pe < s_content_start and \
              current_pe < len(target_text) and \
              (target_text[current_pe] in punctuation_all or target_text[current_pe].isspace()):
            current_pe += 1
        prefix_end_original = current_pe

        # Adjust suffix_start_original to expand leftwards for punctuation/space
        # Stop if we hit the (newly adjusted) end of the prefix (prefix_end_original)
        current_ss = s_content_start
        while current_ss > prefix_end_original and \
              current_ss > 0 and \
              (target_text[current_ss - 1] in punctuation_all or target_text[current_ss - 1].isspace()):
            current_ss -= 1
        suffix_start_original = current_ss
        
        # Ensure prefix_end_original does not exceed suffix_start_original after adjustments.
        # This could happen if the part between p_content_end and s_content_start was all punc/space.
        if prefix_end_original > suffix_start_original:
            prefix_end_original = suffix_start_original

    # Initial segmentation based on adjusted boundaries
    adjusted_target_prefix = target_text[:prefix_end_original]
    adjusted_target_middle = target_text[prefix_end_original:suffix_start_original]
    adjusted_target_suffix = target_text[suffix_start_original:]

    # Helper function to check if a segment is purely punctuation
    def _is_segment_purely_punctuation(segment_text):
        if not segment_text:
            return False
        # punctuation_all is accessible from the outer scope (module global)
        for char_in_segment in segment_text:
            if char_in_segment not in punctuation_all:
                return False
        return True

    # If the identified prefix is purely punctuation, merge it into the middle
    if _is_segment_purely_punctuation(adjusted_target_prefix):
        adjusted_target_middle = adjusted_target_prefix + adjusted_target_middle
        adjusted_target_prefix = ""

    # If the identified suffix is purely punctuation, merge it into the middle
    # This is done after potential prefix adjustment
    if _is_segment_purely_punctuation(adjusted_target_suffix):
        adjusted_target_middle = adjusted_target_middle + adjusted_target_suffix
        adjusted_target_suffix = ""

    return {
        "time_frame": (diff_start_time, diff_end_time),
        "target_prefix": adjusted_target_prefix,
        "target_middle": adjusted_target_middle,
        "target_suffix": adjusted_target_suffix
    }


if __name__ == "__main__":
    prompt_text_zh = "但细细观察便能发现层叠精美，而若影若现的手工刺绣。"
    target_text_zh = "但细细观察便能发现层叠精美，而若影若现的手工刺绣隐约可见的。"
    with open("/home/zhisheng/VoiceCraftX/inference/demo/mfa_alignments/00005418-00000033.csv") as f:
        data_zh = [line.split(",")[:3] for line in f.readlines() if "words" in line]
    alignments_zh = [{"Begin": float(begin), "End": float(end), "Label": label} for begin, end, label in data_zh]

    result_zh = get_diff_time_frame_and_segment(prompt_text_zh, target_text_zh, alignments_zh, clean=True, language="chinese")
    if result_zh:
        end_time, start_time = result_zh["time_frame"]
        print("\nChinese alignment result:")
        print("Time frame of the different part: from {} seconds to {} seconds".format(end_time, start_time))
        print("target_text segmentation result:")
        print("Prefix:", result_zh["target_prefix"])
        print("Middle:", result_zh["target_middle"])
        print("Suffix:", result_zh["target_suffix"])

    # prompt_text_en = "When the carpet and the curtains caught fire, it was getting warm."
    # target_text_en = "When the carpet and the curtains caught fire, the situation became dire."
    # with open("/home/zhisheng/VoiceCraftX/inference/demo/mfa_alignments/common_voice_en_17662945-common_voice_en_17662954.csv", encoding="utf-8") as f:
    #     data_en = [line.split(",")[:3] for line in f.readlines() if "words" in line]
    # alignments_en = [{"Begin": float(begin), "End": float(end), "Label": label} for begin, end, label in data_en]

    # result_en = get_diff_time_frame_and_segment(prompt_text_en, target_text_en, alignments_en, clean=True, language="english")
    # if result_en:
    #     end_time, start_time = result_en["time_frame"]
    #     print("\nEnglish alignment result:")
    #     print("Time frame of the different part: from {} seconds to {} seconds".format(end_time, start_time))
    #     print("target_text segmentation result:")
    #     print("Prefix:", result_en["target_prefix"])
    #     print("Middle:", result_en["target_middle"])
    #     print("Suffix:", result_en["target_suffix"])
