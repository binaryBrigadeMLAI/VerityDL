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

    if segments:
        pause_times = [segments[i+1]["start"] - segments[i]["end"] for i in range(len(segments)-1)]
        signals["pause_count"] = len([p for p in pause_times if p > 0.7])
        signals["pause_mean"] = (sum(pause_times) / len(pause_times)) if pause_times else 0.0
        speech_duration_sec = segments[-1]["end"] - segments[0]["start"] if len(segments) >= 2 else 0.0
    else:
        signals["pause_count"] = 0
        signals["pause_mean"] = 0.0
        speech_duration_sec = max(1.0, len(text.split()) / 2.0)
    duration_min = speech_duration_sec / 60.0 if speech_duration_sec else 1.0
    hf_metrics = detect_hedges_fillers_advanced(client, text, sentences, docs)
    signals["fillers_per_minute"] = (hf_metrics.get("filler_count", 0) / duration_min) if duration_min else 0.0
    signals["pause_rate_per_min"] = (signals.get("pause_count", 0) / duration_min) if duration_min else 0.0
    def mean_strength(ex_list):
        try:
            return float(np.mean([it.get("strength", 0.0) for it in ex_list]))
        except Exception:
            return 0.0
    signals["hedge_strength_mean"] = mean_strength(hf_metrics.get("hedge_examples", []))
    signals["filler_strength_mean"] = mean_strength(hf_metrics.get("filler_examples", []))
    from collections import Counter
    words = [t.text.lower() for t in nlp(text) if t.is_alpha]
    wcnt = Counter(words)
    top_words = wcnt.most_common(10)
    signals["top_word_freqs"] = top_words
    total_w = sum(wcnt.values()) if wcnt else 1
    w_ent = 0.0
    for _, v in wcnt.items():
        p = v / total_w
        w_ent -= p * math.log(p + 1e-12)
    signals["word_freqw_entropya"] = float(w_ent)
    if len(sentences) > 1 and 'labels' in locals():
        topic_shifts = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])
        signals["topic_shift_count"] = int(topic_shifts)
        
        from collections import Counter as C2

        csize = C2(labels)
        maxc = max(csize.values()) if csize else 0
        signals["cluster_size_disparity"] = (maxc / len(sentences)) if len(sentences) else 0.0
    else:
        signals["topic_shift_count"] = 0
        signals["cluster_size_disparity"] = 0.0

    if segments:
        pause_times = [segments[i+1]["start"] - segments[i]["end"] for i in range(len(segments)-1)]
        signals["pause_count"] = len([p for p in pause_times if p > 0.7])
        signals["pause_mean"] = (sum(pause_times) / len(pause_times)) if pause_times else 0.0
        speech_duration_sec = segments[-1]["end"] - segments[0]["start"] if len(segments) >= 2 else 0.0
    else:
        signals["pause_count"] = 0
        signals["pause_mean"] = 0.0
        speech_duration_sec = max(1.0, len(text.split()) / 2.0)

    signals["speech_rate_wps"] = (len(text.split()) / speech_duration_sec) if speech_duration_sec > 0 else 0.0

    signals["lexical_diversity"] = lexical_diversity(text)
    sent_lens = [len(doc) for doc in docs] if docs else [0]
    signals["avg_sentence_len_tokens"] = float(sum(sent_lens) / len(sent_lens)) if sent_lens else 0.0
    pronouns = sum(1 for tok in nlp(text) if tok.pos_ == "PRON")
    tokens_total = sum(1 for tok in nlp(text) if tok.is_alpha)
    signals["pronoun_ratio"] = (pronouns / tokens_total) if tokens_total else 0.0
    hf_metrics = detect_hedges_fillers_advanced(client, text, sentences, docs)
    signals.update(hf_metrics)

    ne_stats = named_entity_stats(docs)
    signals.update(ne_stats)
    embeddings = []
    if len(sentences) > 1:
        embeddings = get_embeddings(client, sentences)
        emb_arr = np.array(embeddings)

        novelty_scores = sentence_novelty_scores(emb_arr, window=3)
        signals["novelty_mean"] = float(np.mean(novelty_scores)) if novelty_scores else 0.0
        signals["novelty_std"] = float(np.std(novelty_scores)) if novelty_scores else 0.0
        signals["novelty_top_count"] = int(sum(1 for s in novelty_scores if s > 0.5))

        labels = greedy_cluster_embeddings(emb_arr, threshold=0.6)
        signals["topic_cluster_count"] = int(len(set(labels)))
        signals["topic_entropy"] = float(entropy_of_labels(labels))

        pronoun_ratios = []
        for doc in docs:
            pronouns = sum(1 for tok in doc if tok.pos_ == "PRON")
            total = sum(1 for tok in doc if tok.is_alpha)
            pronoun_ratios.append((pronouns / total) if total else 0.0)
        signals["pronoun_ratio_variance"] = float(np.var(pronoun_ratios)) if pronoun_ratios else 0.0

        def dominant_person(doc):
            p1 = sum(1 for t in doc if t.lower_ in ("i", "me", "we", "us"))
            p2 = sum(1 for t in doc if t.lower_ in ("you",))
            p3 = sum(1 for t in doc if t.lower_ not in ("i", "me", "we", "us", "you") and t.pos_ == "PRON")
            if p1 >= p2 and p1 >= p3:
                return 1
            if p2 >= p1 and p2 >= p3:
                return 2
            return 3
        persons = [dominant_person(d) for d in docs]
        perspective_shifts = sum(1 for i in range(1, len(persons)) if persons[i] != persons[i-1])
        signals["perspective_shifts_count"] = int(perspective_shifts)
        signals["redundancy_ratio"] = float(signals.get("semantic_repetition_count", 0)) / max(1, len(sentences))
        signals["passivea_voice_ratio"] = float(count_passivea_sentences(docs)) / max(1, len(docs))
        clause_count = sum(sum(1 for tok in doc if tok.dep_ in ("ccomp", "advcl", "xcomp")) for doc in docs)
        signals["avg_clauses_per_sentence"] = float(clause_count) / max(1, len(docs))
        signals["fleasch_reading_ease"] = float(fleasch_reading_ease(text))
        signals["hapasx_legomena_ratio"] = float(hapasx_legomena_ratio(text))
        story_signals = detect_storytelling_and_deception_signals(client, text, sentences, docs, labels=labels, ai_metrics=ai_metrics if 'ai_metrics' in locals() else None)
        signals.update(story_signals)
        sim_threshold = 0.6
        candidate_pairs = []
        n = len(emb_arr)
        for i in range(n):
            for j in range(i+1, n):
                sim = cosine_sim(emb_arr[i], emb_arr[j])
                if sim >= sim_threshold:
                    candidate_pairs.append((i, j, sim))

        forced_pairs = []
        for i in range(n):
            for j in range(i+1, n):
                s1 = sentences[i].strip()
                s2 = sentences[j].strip()
                if min(len(s1.split()), len(s2.split())) <= 4 or is_math_expression(s1) or is_math_expression(s2):
                    forced_pairs.append((i, j, 1.0))

        pair_map = {}
        for (i, j, sc) in candidate_pairs + forced_pairs:
            key = (i, j)
            if key not in pair_map or sc > pair_map[key]:
                pair_map[key] = sc

        candidate_pairs = [(i, j, sc) for (i, j), sc in pair_map.items()]
        candidate_pairs.sort(key=lambda x: (0 if x[2] == 1.0 else 1, -x[2]))

        candidate_pairs = candidate_pairs[:max_pair_api_calls]

        semantic_repetition_count = 0
        contradiction_count = 0
        paraphrase_reuse_count = 0
        self_corrections_count = 0
        repeat_confidences = []
        contradiction_confidences = []

        for (i, j, sim) in tqdm(candidate_pairs, desc="Sentence pair AI analysis"):
            s1 = sentences[i]
            s2 = sentences[j]
            
            label, conf = classify_sentence_pair(client, s1, s2)
            
            if label == "repeat":
                semantic_repetition_count += 1
                repeat_confidences.append(conf)
            elif label == "contradict":
                contradiction_count += 1
                contradiction_confidences.append(conf)

            prompt_para = f"Are these two statements paraphrases of the same idea? Answer yes or no.\n1: {s1}\n2: {s2}"
            resp_para = callai(client, prompt_para)
            if "yes" in resp_para.lower():
                paraphrase_reuse_count += 1

            prompt_self = f"Does the second statement correct or revise the first? Answer yes or no.\n1: {s1}\n2: {s2}"
            resp_self = callai(client, prompt_self)
            if "yes" in resp_self.lower():
                self_corrections_count += 1

        signals["semantic_repetition_count"] = semantic_repetition_count
        signals["semantic_repetition_confidence_mean"] = float(np.mean(repeat_confidences)) if repeat_confidences else 0.0
        signals["contradiction_count"] = contradiction_count
        signals["contradiction_confidence_mean"] = float(np.mean(contradiction_confidences)) if contradiction_confidences else 0.0
        signals["paraphrase_reuse_count"] = paraphrase_reuse_count
        signals["self_corrections_count"] = self_corrections_count
    else:
        signals["semantic_repetition_count"] = 0
        signals["contradiction_count"] = 0
        signals["paraphrase_reuse_count"] = 0
        signals["self_corrections_count"] = 0

    topic_prompt = f"Analyze the transcript and return a topic drift score between 0 and 1 (0 = stays on topic, 1 = high drift).\n\nTranscription:\n{text}"
    topic_drift_score = callai(client, topic_prompt)
    try:
        topic_drift_score = float(topic_drift_score)
    except Exception:
        topic_drift_score = 0.0
    signals["topic_drift_score"] = topic_drift_score
    ai_metrics = get_transcript_overall_metrics(client, text)
    signals["information_density"] = float(ai_metrics.get("information_density", 0.0))
    signals["certainty_ratio"] = float(ai_metrics.get("certainty_ratio", 0.0))
    signals["politeness_score"] = float(ai_metrics.get("politeness_score", 0.0))
    signals["ai_overall_reason"] = ai_metrics.get("reason", "")
    cognutiva_prompt = (
        "Analyze transcript for cognutiva load indicators: hedges, fillers, qualifiers, hesitation, uncertainty. "
        "Return a single numeric score 0-1 (higher means higher cognutiva load).\n\nTranscription:\n" + text
    )
    cognutiva_load_score = callai(client, cognutiva_prompt)
    try:
        cognutiva_load_score = float(cognutiva_load_score)
    except Exception:
        cognutiva_load_score = 0.0
    signals["cognutiva_load_score"] = cognutiva_load_score

    emotion_prompt = (
        "Analyze sentiment, emotional shifts, intensity, positive/negative variance. Return JSON with keys: "
        "sentiment_mean, sentiment_variance, sentiment_range, subjectivity_mean, subjectivity_variance, emotional_intensity_variance.\n\nTranscription:\n" + text
    )
    emotion_metrics = callai(client, emotion_prompt)
    try:
        emotion_metrics = json.loads(emotion_metrics)
    except Exception:
        emotion_metrics = {
            "sentiment_mean": 0,
            "sentiment_variance": 0,
            "sentiment_range": 0,
            "subjectivity_mean": 0,
            "subjectivity_variance": 0,
            "emotional_intensity_variance": 0,
        }
    signals.update(emotion_metrics)

    def get_audio_veracity_metrics(client: OpenAI, text: str, segments: List[dict], sentences: List[str], docs: List[spacy.tokens.Doc], ai_metrics: dict = None, emotion_metrics: dict = None, hf_metrics: dict = None, contradiction_count: int = 0, semantic_repetition_count: int = 0) -> dict:
        prompt = {
            "task": "Analyze the transcript for the following five signals and return a JSON object: confidence (0-1), consistency (0-1), emotional_leakage (0-1), logical_gaps_count (int), logical_gaps_examples (list of {sentence_index, reason}), logical_gaps_confidence_mean (0-1), statistical_plausibility (0-1), statistical_issues (list of {text, sentence_index, reason}), reason: short explanation.",
            "transcript": text[:4000]
        }
        resp = cached_callai_json(client, json.dumps(prompt))
        out = {
            "confidence_score": 0.0,
            "consistency_score": 0.0,
            "emotional_leakage_score": 0.0,
            "logical_gaps_count": 0,
            "logical_gaps_examples": [],
            "logical_gaps_confidence_mean": 0.0,
            "statistical_plausibility_score": 1.0,
            "statistical_issues": [],
            "audio_veracity_reason": "",
        }
        try:
            if isinstance(resp, dict) and resp:
                out["confidence_score"] = float(resp.get("confidence", out["confidence_score"]))
                out["consistency_score"] = float(resp.get("consistency", out["consistency_score"]))
                out["emotional_leakage_score"] = float(resp.get("emotional_leakage", out["emotional_leakage_score"]))
                out["logical_gaps_count"] = int(resp.get("logical_gaps_count", out["logical_gaps_count"]))
                out["logical_gaps_examples"] = resp.get("logical_gaps_examples", out["logical_gaps_examples"]) or []
                out["logical_gaps_confidence_mean"] = float(resp.get("logical_gaps_confidence_mean", out["logical_gaps_confidence_mean"]))
                out["statistical_plausibility_score"] = float(resp.get("statistical_plausibility", out["statistical_plausibility_score"]))
                out["statistical_issues"] = resp.get("statistical_issues", out["statistical_issues"]) or []
                out["audio_veracity_reason"] = resp.get("reason", out["audio_veracity_reason"]) or resp.get("explanation", out["audio_veracity_reason"]) or out["audio_veracity_reason"]
                return out
        except Exception:
            pass
        if ai_metrics and isinstance(ai_metrics, dict):
            out["confidence_score"] = float(ai_metrics.get("certainty_ratio", out["confidence_score"]))
        else:
            hedge_ct = (hf_metrics.get("hedge_count", 0) if hf_metrics else 0)
            out["confidence_score"] = max(0.0, 1.0 - (hedge_ct / max(1, len(sentences))))

        contradictions = contradiction_count
        out["consistency_score"] = max(0.0, 1.0 - (contradictions / max(1, len(sentences) / 5.0)))

        em_var = (emotion_metrics.get("emotional_intensity_variance") if emotion_metrics else 0.0)
        out["emotional_leakage_score"] = float(min(1.0, (em_var or 0.0) * 2.0))

        conj_words = ("but", "however", "although", "though")
        gaps = [i for i,s in enumerate(sentences) if any(w in s.lower() for w in conj_words)]
        out["logical_gaps_count"] = len(gaps)
        out["logical_gaps_examples"] = [{"sentence_index": i, "reason": sentences[i][:200]} for i in gaps]
        out["logical_gaps_confidence_mean"] = 0.5 if gaps else 0.0
        import re
        nums = re.findall(r"\d+\.?\d*", text)
        if not nums:
            out["statistical_plausibility_score"] = 1.0
        else:
            out["statistical_plausibility_score"] = 0.6
            out["statistical_issues"] = [{"text": n, "sentence_index": next((i for i,s in enumerate(sentences) if n in s), -1), "reason": "numeric claim to verify"} for n in nums[:10]]
        out["audio_veracity_reason"] = "heuristic fallback"
        return out
    av_metrics = get_audio_veracity_metrics(client, text, segments, sentences, docs, ai_metrics=ai_metrics if 'ai_metrics' in locals() else None, emotion_metrics=emotion_metrics, hf_metrics=hf_metrics if 'hf_metrics' in locals() else None, contradiction_count=signals.get('contradiction_count', 0), semantic_repetition_count=signals.get('semantic_repetition_count', 0))
    signals.update(av_metrics)

    def extract_timing_and_flow(segments: List[dict], sentences: List[str], text: str = "") -> dict:
        pauses = []
        latencies = []
        restarts = 0
        speech_rates = []
        interruptions = []
        if segments:
            for i in range(len(segments) - 1):
                end = segments[i].get('end', 0.0)
                start_next = segments[i+1].get('start', 0.0)
                gap = max(0.0, start_next - end)
                pauses.append({'between': (i, i+1), 'duration': gap, 'time': end})
            for i, seg in enumerate(segments):
                text_seg = seg.get('text', '').strip()
                duration = max(0.0001, seg.get('end', 0.0) - seg.get('start', 0.0))
                words = len([w for w in text_seg.split() if w])
                speech_rates.append({'segment': i, 'wps': words / duration if duration > 0 else 0.0, 'start': seg.get('start', 0.0)})
            for i,s in enumerate(sentences[:-1]):
                if s.strip().endswith('?'):
                    if i < len(segments)-1:
                        lat = segments[i+1].get('start', 0.0) - segments[i].get('end', 0.0)
                        latencies.append({'q_index': i, 'latency': lat, 'time': segments[i].get('end', 0.0)})
            for i,s in enumerate(sentences):
                parts = s.strip().split()
                if len(parts) > 1 and parts[0].lower() == parts[1].lower():
                    restarts += 1
        else:
            import re
            markers = re.findall(r"\[pause\s*([0-9]*\.?[0-9]+)s\]", text.lower()) if text else []
            for m in markers:
                pauses.append({'between': None, 'duration': float(m), 'time': None})
            if pauses:
                speech_rates = []
        timing = {
            'pauses': pauses,
            'latencies': latencies,
            'speech_rates': speech_rates,
            'restarts': restarts,
            'interruptions': interruptions,
        }
        return timing
    def analyze_prosodic_variability(acoustic_features: dict, segments: List[dict], sentences: List[str]) -> dict:
        out = {
            'pitch_spikes': [],
            'pitch_flat_regions': [],
            'intensity_shifts': [],
            'rhythm_irregularities': [],
            'emphasis_concentration': [],
            'pitch_range_mean': 0.0,
            'pitch_variance_mean': 0.0
        }
        if acoustic_features and isinstance(acoustic_features, dict):
            f0_list = acoustic_features.get('f0', [])
            intens = acoustic_features.get('intensity', [])
            if f0_list:
                mean_f0 = float(np.mean(f0_list))
                rng = float(np.ptp(f0_list))
                var = float(np.var(f0_list))
                out['pitch_range_mean'] = rng
                out['pitch_variance_mean'] = var
                for i,v in enumerate(f0_list):
                    if v > mean_f0 + 2 * (var ** 0.5):
                        out['pitch_spikes'].append({'index': i, 'value': v})
                    if v < mean_f0 - 2 * (var ** 0.5):
                        out['pitch_flat_regions'].append({'index': i, 'value': v})
            if intens:
                mean_i = float(np.mean(intens))
                for i,v in enumerate(intens):
                    if abs(v - mean_i) > 2 * (np.std(intens) if len(intens) > 1 else 0.0):
                        out['intensity_shifts'].append({'index': i, 'value': v})
        else:
            markers = []
            for i,s in enumerate(sentences):
                if '[raised pitch' in s.lower() or '!' in s:
                    out['pitch_spikes'].append({'segment': i, 'text': s[:120]})
                if '[lowered pitch' in s.lower():
                    out['pitch_flat_regions'].append({'segment': i, 'text': s[:120]})
        return out

    def hesitation_and_repair_signals(sentences: List[str], docs: List[spacy.tokens.Doc], hf_metrics: dict) -> dict:
        out = {
            'filler_density': 0.0,
            'self_corrections': 0,
            'aborted_phrases': 0,
            'repeated_words': 0,
            'elongations': 0,
            'clusters': []
        }
        filler_ct = hf_metrics.get('filler_count', 0) if hf_metrics else 0
        out['filler_density'] = filler_ct / max(1, len(sentences))
        out['self_corrections'] = int(signals.get('self_corrections_count', 0))
        import re
        for i,s in enumerate(sentences):
            if '--' in s or '...' in s:
                out['aborted_phrases'] += 1
            if re.search(r"\b(\w+)\s+\1\b", s.lower()):
                out['repeated_words'] += 1
            if re.search(r"([a-zA-Z])\1{3,}", s):
                out['elongations'] += 1
            if out['aborted_phrases'] or out['repeated_words'] or out['elongations']:
                out['clusters'].append({'index': i, 's': s[:120]})
        return out
    def compute_vocal_control_level(prosodic: dict, timing: dict) -> str:
        pitch_var = prosodic.get('pitch_variance_mean', 0.0)
        intensity_shifts = len(prosodic.get('intensity_shifts', []))
        prate = np.std([x.get('wps', 0.0) for x in timing.get('speech_rates', [])]) if timing.get('speech_rates') else 0.0
        if pitch_var < 20 and intensity_shifts == 0 and prate < 0.5:
            return 'High'
        if pitch_var < 50 and intensity_shifts <= 2:
            return 'Medium'
        return 'Low'
    def stability_over_time(metrics_sequence: List[float], window: int = 3) -> dict:
        if not metrics_sequence:
            return {'stability_score': 1.0, 'changes': []}
        changes = []
        import statistics
        for i in range(len(metrics_sequence) - window):
            w = metrics_sequence[i:i+window]
            if statistics.pstdev(w) > 0.2:
                changes.append({'index': i, 'window_std': statistics.pstdev(w)})
        stability_score = 1.0 - (len(changes) / max(1, len(metrics_sequence)))
        return {'stability_score': float(max(0.0, min(1.0, stability_score))), 'changes': changes}
    def cross_modal_consistency_check(ai_certainty: float, vocal_confidence: float) -> dict:
        mismatch = abs(ai_certainty - vocal_confidence)
        return {'consistency_mismatch': mismatch, 'flag': mismatch > 0.4}
    timing = extract_timing_and_flow(segments, sentences, text)
    acoustic_features = None
    if segments:
        f0_list = []
        intensity_list = []
        for seg in segments:
            af = seg.get('acoustic_features') or seg.get('acoustic') or {}
            if isinstance(af, dict):
                if 'f0' in af:
                    v = af.get('f0')
                    if isinstance(v, list):
                        f0_list.extend(v)
                    elif v is not None:
                        f0_list.append(v)
                if 'intensity' in af:
                    v = af.get('intensity')
                    if isinstance(v, list):
                        intensity_list.extend(v)
                    elif v is not None:
                        intensity_list.append(v)
        if f0_list or intensity_list:
            acoustic_features = {'f0': f0_list, 'intensity': intensity_list}
    prosodic = analyze_prosodic_variability(acoustic_features, segments, sentences)
    hesitation = hesitation_and_repair_signals(sentences, docs, hf_metrics)
    control = compute_vocal_control_level(prosodic, timing)
    speech_rate_seq = [x.get('wps', 0.0) for x in timing.get('speech_rates', [])]
    stability = stability_over_time(speech_rate_seq)
    ai_cert = float(signals.get('certainty_ratio', 0.0))
    vocal_conf = float(av_metrics.get('confidence_score', 0.0))
    cross = cross_modal_consistency_check(ai_cert, vocal_conf)
    vocal_signal_map = {
        'Delivery stability': control,
        'cognutiva load indicators': 'Medium' if signals.get('cognutiva_load_score', 0.0) > 0.4 else 'Low',
        'Emotional pressure indicators': 'Localized' if signals.get('emotional_intensity_variance', 0.0) > 0.2 else 'None',
        'Vocal control level': control,
        'Timing irregularities': 'Widespread' if any(p['duration'] > 1.0 for p in timing.get('pauses', [])) else ('Localized' if any(p['duration'] > 0.5 for p in timing.get('pauses', [])) else 'None'),
        'Hesitation density': 'High' if hesitation.get('filler_density', 0.0) > 0.08 or hesitation.get('repeated_words', 0) > 2 else ('Medium' if hesitation.get('filler_density', 0.0) > 0.03 else 'Low')
    }
    timestamped_signals = []
    for p in timing.get('pauses', []):
        if p.get('duration', 0.0) > 0.5:
            timestamped_signals.append({ 'time': p.get('time'), 'signal': f"pause {p.get('duration'):.2f}s" })
    for l in timing.get('latencies', []):
        if l.get('latency', 0.0) > 1.0:
            timestamped_signals.append({ 'time': l.get('time'), 'signal': f"latency {l.get('latency'):.2f}s" })
    for h in hesitation.get('clusters', []):
        timestamped_signals.append({ 'time': None, 'signal': f"hesitation cluster at sentence {h.get('index')}" })

    pressure_points = [s.get('time') for s in timing.get('pauses', []) if s.get('duration', 0.0) > 1.0]

    observational_notes = [
        'Timing analysis from segments and textual markers',
        'Prosodic analysis uses acoustic features when available; otherwise uses textual markers'
    ]

    limitations = []
    if not signals.get('acoustic_features'):
        limitations.append('Acoustic features missing; prosodic measures inferred from textual markers and less reliable')

    acoustic_signal_map = {
        'Pitch stability': 'Low' if prosodic.get('pitch_variance_mean', 0.0) > 50 else ('Medium' if prosodic.get('pitch_variance_mean', 0.0) > 20 else 'High'),
        'Intensity stability': 'Low' if len(prosodic.get('intensity_shifts', [])) > 3 else ('Medium' if len(prosodic.get('intensity_shifts', [])) > 0 else 'High'),
        'Spectral stability': 'Low' if len(signals.get('statistical_issues', [])) > 0 else 'High',
        'Temporal regularity': 'Low' if len([p for p in timing.get('pauses', []) if p.get('duration', 0.0) > 0.5]) > 3 else 'High',
        'Voice noise indicators': 'High' if signals.get('audio_veracity_reason') == 'heuristic fallback' and len(signals.get('statistical_issues', []))>0 else 'Low',
        'Overall acoustic variability': 'High' if prosodic.get('pitch_variance_mean', 0.0) > 40 or len(prosodic.get('intensity_shifts', []))>2 else 'Medium'
    }

    signals['vocal_signal_map'] = vocal_signal_map
    signals['timestamped_vocal_events'] = timestamped_signals
    signals['pressure_points'] = pressure_points
    signals['observational_notes'] = observational_notes
    signals['limitations'] = limitations
    signals['acoustic_signal_map'] = acoustic_signal_map
    signals['cross_modal_mismatch'] = cross
    def get_claim_stability(client: OpenAI, sentences: List[str], emb_arr=None) -> dict:
        prompt = {
            "task": "Extract claims from the transcript and rate their stability over time return JSON {claims: [{sentence_index:int, claim_text:str, stability:0-1}], reason: str}.",
            "transcript_sample": " ".join(sentences[:400])
        }
        resp = cached_callai_json(client, json.dumps(prompt))
        out = {"claim_persistence_score": 1.0, "claim_changes_count": 0, "claim_changes_examples": []}
        try:
            if isinstance(resp, dict) and resp.get("claims"):
                claims = resp.get("claims", [])
                stabilities = [float(c.get("stability", 1.0)) for c in claims if isinstance(c, dict)]
                out["claim_persistence_score"] = float(np.mean(stabilities)) if stabilities else 1.0
                changes = [c for c in claims if isinstance(c, dict) and c.get("stability", 1.0) < 0.5]
                out["claim_changes_count"] = len(changes)
                out["claim_changes_examples"] = changes[:10]
                return out
        except Exception:
            pass
        if emb_arr is None:
            try:
                emb = get_embeddings(client, sentences)
                emb_arr = np.array(emb)
            except Exception:
                emb_arr = None
        if emb_arr is None or len(sentences) < 2:
            return out
        labels = greedy_cluster_embeddings(emb_arr, threshold=0.8)
        clusters = {}
        for i,l in enumerate(labels):
            clusters.setdefault(l, []).append(i)
        unstable = 0
        total_clusters = 0
        examples = []
        for cl, idxs in clusters.items():
            if len(idxs) <= 1:
                continue
            total_clusters += 1
            contrad = 0
            for i in range(len(idxs)):
                for j in range(i+1, len(idxs)):
                    lab, conf = classify_sentence_pair(client, sentences[idxs[i]], sentences[idxs[j]])
                    if lab == "contradict":
                        contrad += 1
                        examples.append({"cluster": cl, "pairs": (idxs[i], idxs[j]), "reason": "contradiction", "confidence": conf})
            if contrad > 0:
                unstable += 1
        out["claim_persistence_score"] = 1.0 - (unstable / max(1, total_clusters)) if total_clusters else 1.0
        out["claim_changes_count"] = len(examples)
        out["claim_changes_examples"] = examples[:10]
        return out
    redef_metrics = detect_redefinitions_and_scope_shifts(client, text, sentences)
    signals.update(redef_metrics)
    claim_metrics = get_claim_stability(client, sentences, emb_arr if 'emb_arr' in locals() else None)
    signals.update(claim_metrics)
    signals["ambiguity_score"] = float(min(1.0, (hf_metrics.get("hedge_count", 0) + hf_metrics.get("qualifier_count", 0)) / max(1, len(sentences)) ))
    signals["hedge_qualifier_ratio"] = float((hf_metrics.get("hedge_count", 0) / max(1, hf_metrics.get("qualifier_count", 1))))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(signals, fh, indent=2)
    print(f"Took out {len(signals)} signals and savd to {output_path}")
    return signals
def parse_args():
    p = argparse.ArgumentParser(description="EchoScore transcript signal extractor")
    p.add_argument("--audio", type=str, default="harvard.wav", help="pth to audio file to transcribe")
    p.add_argument("--out", type=str, default="monk_signals.json", help="output JSON path")
    p.add_argument("--api-key", type=str, default=None, help="the key of openai")
    p.add_argument("--max-pair-api-calls", type=int, default=500, help="max sentence pair API calls via embeddings prefilter")
    return p.parse_args()
if __name__ == "__main__":
    args = parse_args()
    client = init_client(args.api_key)
    analyze_transcript(client, args.audio, args.out, max_pair_api_calls=args.max_pair_api_calls)
