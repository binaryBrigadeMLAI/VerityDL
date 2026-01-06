# This is the big old beautiful heart of the EchoScore algorithm

import os
import time
import math
import json
import argparse
from pathlib import Path
from typing import List, Tuple
from spacy.matcher import Matcher
import numpy as np
from tqdm import tqdm
import spacy
from openai import OpenAI
import hashlib
import pickle
try:
    import sympy as sp
except Exception:
    sp = None
nlp = spacy.load("en_core_web_sm")
def init_client(api_key: str = None) -> OpenAI:
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("no openai key arnav bru lock in")
    return OpenAI(api_key=key)
def retry_backpoff(fn, *args, retries=4, base_delay=1, **kwargs):
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))
def load_audio_transcript(client: OpenAI, audio_path: str) -> Tuple[str, List[dict]]:
    with open(audio_path, "rb") as fh:
        resp = retry_backpoff(
            client.audio.transcriptions.create,
            model="whisper-1",
            file=fh,
            response_format="verbose_json",
        )

    text = resp.get("text", "")
    segments = resp.get("segments", [])
    return text, segments
def callai(client: OpenAI, prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.0) -> str:
    resp = retry_backpoff(
        client.chat.completions.create,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    try:
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""
def callai_json(client: OpenAI, prompt: str, **kwargs) -> dict:
    txt = callai(client, prompt, **kwargs)
    try:
        return json.loads(txt)
    except Exception:
        return {}
def named_entity_stats(docs: List[spacy.tokens.Doc]) -> dict:
    from collections import Counter

    ents = []
    for d in docs:
        ents.extend([(e.label_, e.text) for e in d.ents])
    if not ents:
        return {"named_entity_count": 0, "unique_entity_types": 0, "top_entities": []}
    types = [t for t, _ in ents]
    cnt = Counter(types)
    top = cnt.most_common(5)
    return {"named_entity_count": len(ents), "unique_entity_types": len(cnt), "top_entities": top}
def get_transcript_overall_metrics(client: OpenAI, text: str) -> dict:
    prompt = (
        "Analyze the transcript and return a JSON object with the following keys: \n"
        "information_density: number 0-1 (how packed with novel, relevant info), \n"
        "certainty_ratio: number 0-1 (fraction of strongly asserted statements vs hedged), \n"
        "politeness_score: number 0-1 (higher = more polite), \n"
        "and a short explanation field 'reason'\n\nTranscript:\n" + text[:8000]
    )
    return callai_json(client, prompt)
def get_embeddings(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    resp = retry_backpoff(client.embeddings.create, input=texts, model=model)
    return [d.embedding for d in resp.data]
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def is_math_expression(s: str) -> bool:
    """Return True if string looks like a math expression (contains digits and operators)"""
    s = s.strip()
    allowed = set("0123456789+-*/=(). ^")
    return any(ch.isdigit() for ch in s) and any(ch in allowed for ch in s)
def normalize_math_equation(s: str) -> Tuple[bool, float]:
    """Try to parse an equation like '1+1=2' and return (is_equation, difference)

    If parsing fails or SymPy not available, returns (False, 0.0)
    """
    if sp is None:
        return False, 0.0
    if "=" not in s:
        return False, 0.0
    try:
        left, right = s.split("=", 1)
        L = sp.sympify(left)
        R = sp.sympify(right)
        diff = sp.simplify(L - R)
        is_zero = diff == 0
        return True, float(0.0 if is_zero else 1.0)
    except Exception:
        return False, 0.0

_AI_CACHE_PATH = Path.home().joinpath(".monk_ai_cache.pkl")
try:
    with open(_AI_CACHE_PATH, "rb") as _fh:
        _AI_CACHE = pickle.load(_fh)
except Exception:
    _AI_CACHE = {}
def _save_ai_cache():
    try:
        with open(_AI_CACHE_PATH, "wb") as fh:
            pickle.dump(_AI_CACHE, fh)
    except Exception:
        pass

def cached_callai_json(client: OpenAI, prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.0) -> dict:
    key = hashlib.sha256((model + "|" + prompt).encode("utf-8")).hexdigest()
    if key in _AI_CACHE:
        return _AI_CACHE[key]
    txt = callai(client, prompt, model=model, temperature=temperature)
    try:
        out = json.loads(txt)
    except Exception:
        out = {"_raw": txt}
    _AI_CACHE[key] = out
    _save_ai_cache()
    return out


def classify_sentence_pair(client: OpenAI, s1: str, s2: str) -> Tuple[str, float]:
    """Return (label, confidence) where label in {repeat, contradict, other} and confidence 0-1

    Uses math normalization and cached AI JSON classification
    """
    for s in (s1, s2):
        if is_math_expression(s):
            ok, diff = normalize_math_equation(s)
            if ok and diff == 0.0:
                return "repeat", 1.0

    prompt = {
        "pair": {"a": s1, "b": s2},
        "task": "Classify whether these two phrases repeat the same idea, contradict, or neither. Return JSON {label: 'repeat'|'contradict'|'other', confidence: 0-1, reason: 'one sentence'}"
    }
    resp = cached_callai_json(client, json.dumps(prompt))
    if not resp:
        return "other", 0.0
    label = resp.get("label") or resp.get("result") or ("repeat" if "yes" in resp.get("_raw", "").lower() else "other")
    conf = float(resp.get("confidence", 0.0) or resp.get("score", 0.0) or 0.0)
    return label, conf
def syllable_count(word: str) -> int:
    w = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_was_vowel = False
    for ch in w:
        is_vowel = ch in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel
    if w.endswith("e") and count > 1:
        count -= 1
    if count == 0:
        count = 1
    return count
def fleasch_reading_ease(text: str) -> float:
    words = [w for w in text.split() if any(c.isalpha() for c in w)]
    sentences = [s for s in text.split(".") if s.strip()]
    syllables = sum(syllable_count(w) for w in words)
    num_words = max(1, len(words))
    num_sentences = max(1, len(sentences))
    asl = num_words / num_sentences
    asw = syllables / num_words
    score = 206.835 - 1.015 * asl - 84.6 * asw
    return float(score)
def count_passivea_sentences(docs: List[spacy.tokens.Doc]) -> int:
    c = 0
    for doc in docs:
        if any(tok.dep_ == "nsubjpass" for tok in doc):
            c += 1
    return c
def hapasx_legomena_ratio(text: str) -> float:
    toks = [t.text.lower() for t in nlp(text) if t.is_alpha]
    if not toks:
        return 0.0
    from collections import Counter

    cnt = Counter(toks)
    hapax = sum(1 for v in cnt.values() if v == 1)
    return hapax / len(toks)


def greedy_cluster_embeddings(emb_arr: np.ndarray, threshold: float = 0.6) -> List[int]:
    if len(emb_arr) == 0:
        return []
    centroids = [emb_arr[0].astype(float)]
    labels = [0]
    for i in range(1, len(emb_arr)):
        v = emb_arr[i].astype(float)
        sims = [cosine_sim(v, c) for c in centroids]
        best = max(sims)
        if best >= threshold:
            idx = sims.index(best)
            centroids[idx] = (centroids[idx] + v) / 2.0
            labels.append(idx)
        else:
            centroids.append(v)
            labels.append(len(centroids) - 1)
    return labels
def entropy_of_labels(labels: List[int]) -> float:
    if not labels:
        return 0.0
    from collections import Counter

    cnt = Counter(labels)
    total = sum(cnt.values())
    ent = 0.0
    for v in cnt.values():
        p = v / total
        ent -= p * math.log(p + 1e-12)
    k = len(cnt)
    if k <= 1:
        return 0.0
    return ent / math.log(k)
def sentence_novelty_scores(emb_arr: np.ndarray, window: int = 3) -> List[float]:
    n = len(emb_arr)
    if n == 0:
        return []
    scores = []
    for i in range(n):
        start = max(0, i - window)
        context = [emb_arr[j] for j in range(start, i)]
        if not context:
            scores.append(0.0)
            continue
        sims = [cosine_sim(emb_arr[i], c) for c in context]
        mean_sim = float(sum(sims) / len(sims))
        scores.append(1.0 - mean_sim)
    return scores

def lexical_diversity(text: str) -> float:
    toks = [t.text.lower() for t in nlp(text) if t.is_alpha]
    if not toks:
        return 0.0
    return len(set(toks)) / len(toks)


def generate_candidate_spans(docs: List[spacy.tokens.Doc]) -> List[dict]:
    matcher = Matcher(nlp.vocab)

    matcher.add("MODAL", [[{"TAG": "MD"}]])
    matcher.add("INTERJECTION", [[{"TAG": "UH"}], [{"POS": "INTJ"}]])
    matcher.add("PRON_VERB", [[{"POS": "PRON"}, {"POS": "VERB"}]])
    matcher.add("IT_VERB", [[{"LOWER": "it"}, {"POS": "VERB"}]])
    matcher.add("ADV_QUAL", [[{"POS": "ADV"}], [{"DEP": "advmod"}]])

    candidates = []
    for si, doc in enumerate(docs):
        matches = matcher(doc)
        seen_spans = set()
        for mid, start, end in matches:
            span = doc[start:end]
            key = (si, span.start_char, span.end_char)
            if key in seen_spans:
                continue
            seen_spans.add(key)
            left = max(span.start - 1, 0)
            right = min(span.end + 1, len(doc))
            ext_span = doc[left:right]
            candidates.append(
                {
                    "text": ext_span.text,
                    "sentence_index": si,
                    "context": doc.text,
                    "start_char": ext_span.start_char,
                    "end_char": ext_span.end_char,
                }
            )
    unique = {}
    for c in candidates:
        k = c["text"].strip().lower()
        if not k:
            continue
        if k not in unique:
            unique[k] = c
    return list(unique.values())


def detect_hedges_fillers_advanced(client: OpenAI, text: str, sentences: List[str], docs: List[spacy.tokens.Doc]) -> dict:
    candidates = generate_candidate_spans(docs)
    if not candidates:
        return {
            "hedge_count": 0,
            "filler_count": 0,
            "qualifier_count": 0,
            "defensive_phrases_count": 0,
            "justification_phrases_count": 0,
            "hedge_examples": [],
        }
    max_cands = 200
    candidates = candidates[:max_cands]
    prompt_payload = {
        "candidates": [
            {"text": c["text"], "sentence_index": c["sentence_index"], "context": c["context"]}
            for c in candidates
        ],
        "instructions": (
            "For each candidate phrase, classify it into one of: hedge, filler, qualifier, defensive, justification, or other "
            "Also provide a numeric strength 0-1 (higher = stronger hedge/qualifier/defense) and a one sentence reason "
            "Return a JSON array of objects: {text, sentence_index, label, strength, reason}"
        ),
    }
    prompt = (
        "You are an assistant that specializes in discourse and pragmatics "
        "Given a transcript and a list of candidate phrases that may be hedges, fillers, qualifiers, defensive phrases or justifications, classify each candidate and score its strength between 0 and 1 "
        "Respond ONLY with a JSON array as specified\n\n" + json.dumps(prompt_payload)
    )
    resp = callai(client, prompt)
    classified = []
    try:
        classified = json.loads(resp)
    except Exception:
        classified = []
        for c in candidates:
            txt = c["text"].lower()
            label = "other"
            strength = 0.0
            if any(tok.tag_ == "UH" for tok in nlp(txt)):
                label = "filler"
                strength = 0.9
            elif any(tok.tag_ == "MD" for tok in nlp(txt)):
                label = "hedge"
                strength = 0.8
            classified.append({"text": c["text"], "sentence_index": c["sentence_index"], "label": label, "strength": strength, "reason": "heuristic fallback"})
    counts = {"hedge": 0, "filler": 0, "qualifier": 0, "defensive": 0, "justification": 0}
    examples = {k: [] for k in counts.keys()}
    for item in classified:
        lbl = item.get("label", "other")
        if lbl in counts:
            counts[lbl] += 1
            examples[lbl].append((item.get("strength", 0.0), item.get("text", ""), item.get("reason", "")))

    for k in examples:
        examples[k].sort(key=lambda x: -float(x[0]))
        examples[k] = [ {"text": t, "strength": s, "reason": r} for (s,t,r) in examples[k][:5] ]

    return {
        "hedge_count": counts["hedge"],
        "filler_count": counts["filler"],
        "qualifier_count": counts["qualifier"],
        "defensive_phrases_count": counts["defensive"],
        "justification_phrases_count": counts["justification"],
        "hedge_examples": examples["hedge"],
        "filler_examples": examples["filler"],
        "qualifier_examples": examples["qualifier"],
    }
def detect_storytelling_and_deception_signals(client: OpenAI, text: str, sentences: List[str], docs: List[spacy.tokens.Doc], labels: List[int] = None, ai_metrics: dict = None) -> dict:
    bridge_phrases = ["so anyway", "anyway", "moving on", "as I said", "like I said", "anyways", "to be honest", "honestly", "you know", "so yeah", "but anyway", "and then"]
    low_sensory_words = set(["see","saw","seen","look","looked","seen","heard","hear","smell","smelled","taste","tasted","felt","feel","feelings","scent","sound","noise","color","colour","texture"])
    text_l = text.lower()
    bridge_count = sum(text_l.count(p) for p in bridge_phrases)
    tokens = [t for t in nlp(text) if t.is_alpha]
    total_alpha = max(1, sum(1 for t in tokens))
    sensory_count = sum(1 for t in tokens if t.lemma_.lower() in low_sensory_words)
    sensory_ratio = sensory_count / total_alpha
    topic_clusters = len(set(labels)) if labels is not None else 0
    sens_norm = min(1.0, sensory_ratio * 5.0)
    bridge_norm = min(1.0, bridge_count / max(1, len(sentences) / 5.0))
    cluster_factor = 1.0 - min(1.0, topic_clusters / 5.0)
    scripted_vague_score = float(min(1.0, (1.0 - sens_norm) * 0.5 + bridge_norm * 0.35 + cluster_factor * 0.15))
    adj_count = sum(1 for t in tokens if t.pos_ == "ADJ")
    adv_count = sum(1 for t in tokens if t.pos_ == "ADV")
    noun_count = max(1, sum(1 for t in tokens if t.pos_ == "NOUN"))
    embellishment_score = float(min(1.0, (adj_count + adv_count) / noun_count / 2.0))
    words_per_sentence = [len(s.split()) for s in sentences] if sentences else [0]
    avg_sentence_len = float(sum(words_per_sentence) / len(words_per_sentence)) if words_per_sentence else 0.0
    too_little_detail = 1.0 if avg_sentence_len < 6.0 or (ai_metrics and float(ai_metrics.get("information_density", 0.0)) < 0.2) else 0.0
    too_much_detail = 1.0 if avg_sentence_len > 25.0 or embellishment_score > 0.9 else 0.0
    first_person = set(["i","me","we","us","my","our","mine","ours"])
    third_person = set(["he","she","they","them","his","her","their","him","hers"])
    fp = sum(1 for t in tokens if t.text.lower() in first_person)
    tp = sum(1 for t in tokens if t.text.lower() in third_person)
    distancing_score = float(tp / max(1, fp + tp))
    question_sents = [s.strip() for s in sentences if s.strip().endswith("?")]
    q_norm = [q.lower() for q in question_sents]
    dup_questions = sum(1 for i,q in enumerate(q_norm) for j in range(i+1, len(q_norm)) if q == q_norm[j])
    repeating_question_count = dup_questions
    if len(question_sents) > 1:
        try:
            q_emb = get_embeddings(client, question_sents)
            q_arr = np.array(q_emb)
            for i in range(len(q_arr)):
                for j in range(i+1, len(q_arr)):
                    if cosine_sim(q_arr[i], q_arr[j]) > 0.88:
                        repeating_question_count += 1
        except Exception:
            pass
    return {
        "scripted_vague_score": scripted_vague_score,
        "sensory_detail_ratio": sensory_ratio,
        "bridge_phrase_count": bridge_count,
        "embellishment_score": embellishment_score,
        "too_little_detail": too_little_detail,
        "too_much_detail": too_much_detail,
        "distancing_score": distancing_score,
        "repeating_question_count": repeating_question_count,
    }
def analyze_transcript(client: OpenAI, audio_path: str, output_path: str, max_pair_api_calls: int = 500):
    text, segments = load_audio_transcript(client, audio_path)
    sentences = [s.get("text", "") for s in segments] if segments else [s.strip() for s in text.split(".") if s.strip()]
    signals = {}
    signals["sentence_count"] = len(sentences)
    docs = list(nlp.pipe(sentences))
    signals["token_count"] = sum(len(doc) for doc in docs)
    signals["question_count"] = sum(1 for s in sentences if s.strip().endswith("?"))
    signals["question_ratio"] = signals["question_count"] / max(1, signals["sentence_count"])
    pos_counts = {"NOUN": 0, "VERB": 0, "ADJ": 0, "ADV": 0, "PRON": 0}
    stopwords = nlp.Defaults.stop_words
    func_words = 0
    total_alpha = 0
    for tok in nlp(text):
        if tok.is_alpha:
            total_alpha += 1
            if tok.lower_ in stopwords:
                func_words += 1
        if tok.pos_ in pos_counts:
            pos_counts[tok.pos_] += 1
    signals["noun_ratio"] = (pos_counts["NOUN"] / total_alpha) if total_alpha else 0.0
    signals["verb_ratio"] = (pos_counts["VERB"] / total_alpha) if total_alpha else 0.0
    signals["adj_ratio"] = (pos_counts["ADJ"] / total_alpha) if total_alpha else 0.0
    signals["adv_ratio"] = (pos_counts["ADV"] / total_alpha) if total_alpha else 0.0
    signals["function_word_ratio"] = (func_words / total_alpha) if total_alpha else 0.0

    signals["exclamation_count"] = text.count("!")
    signals["comma_count"] = text.count(",")
    signals["ellipsis_count"] = text.count("...")

# Still working on rest.
