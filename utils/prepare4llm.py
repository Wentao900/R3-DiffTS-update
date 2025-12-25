from pathlib import Path

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
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

def get_llm(llm_model:str, llm_layers:int=0):
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
