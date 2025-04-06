import re
import time
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import spacy
import pandas as pd
import matplotlib.pyplot as plt

nlp = spacy.load("fi_core_news_sm")

tokenizer = GPT2Tokenizer.from_pretrained("Finnish-NLP/gpt2-finnish")
model = GPT2LMHeadModel.from_pretrained("Finnish-NLP/gpt2-finnish")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

iteraatio_maara = 10
tulostiedosto = Path("novellit.txt")
mittaritiedosto = Path("mittarit.csv")

if tulostiedosto.exists():
    with open(tulostiedosto, "r", encoding="utf-8") as f:
        raw_text = f.read()
        
    normalized_text = re.sub(r"[^a-zA-ZåäöÅÄÖ\s.,!?]", "", raw_text)
    with open(tulostiedosto, "w", encoding="utf-8") as f:
        f.write(normalized_text)

def generoi_novelli(pituus_tokenina=800):
    try:
        with open(tulostiedosto, "r", encoding="utf-8") as f:
            aineisto = f.read()
            prompti = aineisto[-1500:] if len(aineisto) > 1500 else aineisto
    except FileNotFoundError:
        prompti = ""

    prompti += "\n\nTähän haluamasi kehote."

    input_ids = tokenizer.encode(
        prompti,
        return_tensors="pt",
        truncation=True,
        max_length=512 #Tämän voi pidentää 1024-pituiseksi halutessaan. 
    )

    output = model.generate(
        input_ids=input_ids,
        max_length=pituus_tokenina,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.9,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    novelli_ilman_promptia = generated.replace(prompti, "").strip() if prompti in generated else generated
    return generated, novelli_ilman_promptia

def analysoi_teksti(teksti):
    doc = nlp(teksti)
    lauseet = list(doc.sents)
    lauseiden_maara = len(lauseet)
    sanat = [token.text.lower() for token in doc if token.is_alpha]
    uniikit_sanat = len(set(sanat))
    kokonaissanat = len(sanat)
    keskipituus = (
        sum(len(lause) for lause in lauseet) / lauseiden_maara
        if lauseiden_maara > 0 else 0
    )
    ttr = uniikit_sanat / kokonaissanat if kokonaissanat > 0 else 0
    return uniikit_sanat, lauseiden_maara, keskipituus, ttr

if not tulostiedosto.exists():
    tulostiedosto.write_text("", encoding="utf-8")

if not mittaritiedosto.exists():
    mittaritiedosto.write_text("Iteraatio,UniikitSanat,Lauseita,KeskimPituus,TTR\n", encoding="utf-8")

for i in range(1, iteraatio_maara + 1):
    print(f"\n\U0001f300 Iteraatio {i}/{iteraatio_maara}")

    try:
        novelli, analyysiteksti = generoi_novelli()
        uniikit, lauseet, keskipituus, ttr = analysoi_teksti(analyysiteksti)

        print(f"\U0001f4ca Uniikkeja sanoja: {uniikit}, Lauseita: {lauseet}, Keskipit. lause: {keskipituus:.2f} sanaa, TTR: {ttr:.4f}")

        with open(mittaritiedosto, "a", encoding="utf-8") as f:
            f.write(f"{i},{uniikit},{lauseet},{keskipituus:.2f},{ttr:.4f}\n")

        with open(tulostiedosto, "a", encoding="utf-8") as f:
            f.write(f"\n\n### Novelli {i} ###\n")
            f.write(novelli)
            f.write(f"\n\n[Uniikkeja sanoja: {uniikit} | Lauseita: {lauseet} | Keskim. pituus: {keskipituus:.2f} sanaa | TTR: {ttr:.4f}]\n")

    except Exception as e:
        print(f"⚠️ Iteraatio {i}: virhe generoinnissa tai kirjoituksessa: {e}")
        continue

    time.sleep(1)

print("✅ Kaikki novellit generoitu ja tallennettu.")

try:
    df = pd.read_csv(mittaritiedosto)
    df.plot(x="Iteraatio", y=["UniikitSanat", "Lauseita", "KeskimPituus", "TTR"], subplots=True, marker="o", figsize=(10, 8))
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"⚠️ Virhe mittarien visualisoinnissa: {e}")
