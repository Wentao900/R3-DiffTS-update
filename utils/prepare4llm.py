import re
import zlib
from types import SimpleNamespace

from pathlib import Path

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
import torch
import torch.nn as nn
import transformers


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_BERT_PATH = PROJECT_ROOT / "bert-base-uncased"


def get_desc(domain, lookback_len, pred_len):
    
    description = {
        "Agriculture": ["Retail Broiler Composite", "month"],
        "Climate": ["Drought Level", "week"],
        "Economy": ["International Trade Balance", "month"],
        "Energy": ["Gasoline Prices", "week"],
        "Environment": ["Air Quality Index", "day"],
        "Health_US": ["Influenza Patients Proportion", "week"],
        "Security": ["Disaster and Emergency Grants", "month"],
        "SocialGood": ["Unemployment Rate", "month"],
        "Traffic": ["Travel Volume", "month"]
    }

    [OT, freq] = description[domain]

    desc = (f"Below is historical reporting information over the past {lookback_len} {freq}s concerning the {OT}. " 
        f"Based on these reports, predict the potential trends and anomalies of the {OT} for the next {pred_len} {freq}s.")
    return desc

class _StubEncoding(dict):
    """Minimal BatchEncoding: the encoder path calls `.to(device)` on the tokenizer output."""

    def to(self, device):
        moved = _StubEncoding()
        for key, value in self.items():
            moved[key] = value.to(device) if hasattr(value, "to") else value
        return moved


class _StubTokenizer:
    """Hashing tokenizer used when BERT weights are not on disk."""

    def __call__(self, texts, padding="max_length", truncation=True, max_length=32, return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            tokens = re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9\s]", str(text).lower())
            ids = [101]
            for token in tokens:
                if truncation and len(ids) >= max_length - 1:
                    break
                ids.append((zlib.adler32(token.encode("utf-8")) % 30000) + 1)
            ids = ids[:max_length]
            ids = ids + [0] * (max_length - len(ids))
            rows.append(ids)
        input_ids = torch.tensor(rows, dtype=torch.long)
        attention_mask = (input_ids != 0).long()
        if return_tensors == "pt":
            return _StubEncoding(input_ids=input_ids, attention_mask=attention_mask)
        return {"input_ids": input_ids.tolist(), "attention_mask": attention_mask.tolist()}

    def batch_decode(self, input_ids):
        count = input_ids.shape[0] if hasattr(input_ids, "shape") else len(input_ids)
        return [""] * int(count)


class _StubTextEncoder(nn.Module):
    def __init__(self, hidden_size=32, vocab_size=30522):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=int(hidden_size))
        self.embed = nn.Embedding(vocab_size, int(hidden_size))

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.embed(input_ids.clamp(min=0, max=self.embed.num_embeddings - 1))
        return SimpleNamespace(last_hidden_state=hidden)


def get_llm(llm_model:str, llm_layers:int=0):
    if llm_model == "stub":
        return _StubTextEncoder(), _StubTokenizer()
    if llm_model == 'llama':
        # llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
        llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b', local_files_only=True)
        if llm_layers:
            llama_config.num_hidden_layers = llm_layers
        llama_config.output_attentions = True
        llama_config.output_hidden_states = True
        try:
            llm_model = LlamaModel.from_pretrained(
                # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                'huggyllama/llama-7b',
                local_files_only=True,
                config=llama_config,
                # load_in_4bit=True
            )
        except EnvironmentError as exc:
            raise FileNotFoundError("Local LLaMA weights not found; download and place them locally before running.") from exc
        try:
            tokenizer = LlamaTokenizer.from_pretrained(
                # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                'huggyllama/llama-7b',
                local_files_only=True
            )
        except EnvironmentError as exc:
            raise FileNotFoundError("Local LLaMA tokenizer files not found; store them locally before running.") from exc
    elif llm_model == 'gpt2':
        gpt2_config = GPT2Config.from_pretrained('../llms/gpt2', local_files_only=True)
        if llm_layers:
            gpt2_config.num_hidden_layers = llm_layers
        gpt2_config.output_attentions = True
        gpt2_config.output_hidden_states = True
        try:
            llm_model = GPT2Model.from_pretrained(
                'openai-community/gpt2',
                local_files_only=True,
                config=gpt2_config,
            )
        except EnvironmentError as exc:
            raise FileNotFoundError("Local GPT2 weights not found; download and place them locally before running.") from exc

        try:
            tokenizer = GPT2Tokenizer.from_pretrained(
                'openai-community/gpt2',
                local_files_only=True
            )
        except EnvironmentError as exc:
            raise FileNotFoundError("Local GPT2 tokenizer files not found; store them locally before running.") from exc
    elif llm_model == 'bert':
        if not LOCAL_BERT_PATH.exists():
            raise FileNotFoundError(f"Local BERT weights not found at {LOCAL_BERT_PATH}")

        bert_config = BertConfig.from_pretrained(LOCAL_BERT_PATH, local_files_only=True)
        if llm_layers:
            bert_config.num_hidden_layers = llm_layers
        bert_config.output_attentions = True
        bert_config.output_hidden_states = True
        llm_model = BertModel.from_pretrained(
            LOCAL_BERT_PATH,
            local_files_only=True,
            config=bert_config,
        )

        tokenizer = BertTokenizer.from_pretrained(
            LOCAL_BERT_PATH,
            local_files_only=True
        )
    else:
        raise Exception('LLM model is not defined')
    return llm_model, tokenizer
