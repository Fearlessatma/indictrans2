import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from IndicTransToolkit import IndicProcessor

# Constants
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
quantization = None

def initialize_model_and_tokenizer(ckpt_dir, quantization):
    """Initialize the model and tokenizer with optional quantization."""
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig is None:
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()

    model.eval()
    return tokenizer, model


def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    """Batch translate sentences from src_lang to tgt_lang."""
    translations = []
    
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]

        # Preprocess the batch and extract entity mappings
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize the batch and generate input encodings
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        with tokenizer.as_target_tokenizer():
            generated_tokens = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        torch.cuda.empty_cache()

    return translations


# Initialize the model and processor
en_indic_ckpt_dir = "ai4bharat/indictrans2-indic-en-1B"
en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir, quantization)
ip = IndicProcessor(inference=True)

# Sample sentences
en_sents = [
    """അടുത്തിടെയായി എന്റെ ശരീരം വളരെ ക്ഷീണിതവും ക്ഷീണിതവുമായി തോന്നുന്നു. എനിക്ക് ധാരാളം ചുമയുണ്ട്, ചിലപ്പോൾ ചുമയ്ക്കുമ്പോൾ കഫം പുറത്തുവരുന്നു. ചുമയ്ക്കുമ്പോഴോ ദീർഘമായി ശ്വാസം എടുക്കുമ്പോഴോ നെഞ്ചുവേദന വർദ്ധിക്കുന്നു, ഇത് ശ്വസിക്കാൻ അസ്വസ്ഥത ഉണ്ടാക്കുന്നു.

കഴിഞ്ഞ രണ്ട് ദിവസമായി, എന്റെ ശരീര താപനില സാധാരണയേക്കാൾ കൂടുതലാണെന്ന് എനിക്ക് തോന്നുന്നു. രാത്രിയിൽ, ഞാൻ അമിതമായി വിയർക്കുന്നു, എന്നാൽ അതേ സമയം, എനിക്ക് തണുപ്പും വിറയലും അനുഭവപ്പെടുന്നു. ചിലപ്പോൾ, എന്റെ കൈകളും കാലുകളും അസാധാരണമാംവിധം തണുപ്പ് അനുഭവപ്പെടുന്നു, പ്രത്യേകിച്ച് രാവിലെയും രാത്രി വൈകിയും."""
]

# Translation
src_lang, tgt_lang = "mal_Mlym", "eng_Latn"
hi_translations = batch_translate(en_sents, src_lang, tgt_lang, en_indic_model, en_indic_tokenizer, ip)

# Print translations
print(f"\n{src_lang} - {tgt_lang}")
for input_sentence, translation in zip(en_sents, hi_translations):
    print(f"{src_lang}: {input_sentence}")
    print(f"{tgt_lang}: {translation}")

# Free GPU memory
del en_indic_tokenizer, en_indic_model
torch.cuda.empty_cache()
