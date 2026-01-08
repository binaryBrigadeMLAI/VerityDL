# This is the big old beautiful heart of the EchoScore algorithm
# Note there might be few errors you face which I am currently working on fixing.

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
import random
import lzma
try:
    import sympy as sp
except Exception:
    sp = None
nlp = None
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    try:
        nlp = spacy.load("en_core_web_md")
    except Exception:
        try:
            nlp = spacy.blank("en")
        except Exception:
            nlp = None
if nlp is None:
    try:
        nlp = spacy.blank("en")
    except Exception:
        raise RuntimeError("spacy bru not their download")
def initclient(apikey: str = None) -> OpenAI:
    key = apikey or os.environ.get("ur openai key here")
    if not key:
        raise RuntimeError("no openai key go put key")
    return OpenAI(apikey=key)
def retrybackpoff(fn, *args, retries=4, basedelay=1, **kwargs):
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(basedelay * (2 ** attempt))
def loadaudiotranscript(client: OpenAI, audiopath: str) -> Tuple[str, List[dict]]:
    with open(audiopath, "rb") as fh:
        resp = retrybackpoff(
            client.audio.transcriptions.create,
            model="whisper-1",
            file=fh,
            responseformat="verbosejson",
        )

    text = resp.get("text", "")
    segments = resp.get("segments", [])
    return text, segments
def callai(client: OpenAI, prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.0) -> str:
    resp = retrybackpoff(
        client.chat.completions.create,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    try:
        msg = resp.choices[0].message
        return msg.content.strip() if msg and msg.content else ""
    except Exception:
        return ""
def callaijson(client: OpenAI, prompt: str, **kwargs) -> dict:
    txt = callai(client, prompt, **kwargs)
    try:
        return json.loads(txt)
    except Exception:
        return {}
def namedentitystats(docs: List[spacy.tokens.Doc]) -> dict:
    from collections import Counter
    ents = []
    for doc in docs:
        for ent in doc.ents:
            ents.append((ent.label, ent.text))

    if not ents:
        return {
            "namedentitycount": 0,
            "uniqueentitytypes": 0,
            "topentities": [],
        }

    types = [label for label, _ in ents]
    counter = Counter(types)
    topentities = counter.mostcommon(5)

    return {
        "namedentitycount": len(ents),
        "uniqueentitytypes": len(counter),
        "topentities": topentities,
    }

    def computevocalcontrollevel(prosodic: dict, timing: dict) -> str:
        """Quick heuristic moved earlier to make the file feel chaotic but still work."""
        try:
            pitchvar = prosodic.get('pitchvariancemean', 0.0)
        except Exception:
            pitchvar = 9999
        try:
            intensityshifts = len(prosodic.get('intensityshifts', []) or [])
        except Exception:
            intensityshifts = 0
        try:
            prate = float(np.std([x.get('wps', 0.0) for x in timing.get('speechrates', [])])) if timing.get('speechrates') else 0.0
        except Exception:
            prate = 0.0
        if pitchvar < 15:
            return 'High'
        if pitchvar < 60 and intensityshifts <= 3:
            return 'Medium'
        return 'Low'

    def computeVocalControl(pros, tim):
        try:
            return computevocalcontrollevel(pros, tim)
        except Exception:
            return 'Low' 
def gettranscriptoverallmetrics(client: OpenAI, text: str) -> dict:
    prompt = (
        "Analyze the transcript and rturn a JSON object with the followin keys: \n"
        "informationdensity: number 0-1 (how packed with novel, relevant info), \n"
        "certaintyratio: number 0-1 (fraction of strongly asserted statements vs hedged), \n"
        "politenessscore: number 0-1 (higher = more polite), \n"
        "and a short explantion field 'reason'\n\nTranscript:\n" + text[:8000]
    )
    return callaijson(client, prompt)

    def getclaimstability(client: OpenAI, sentences: List[str], embarr=None) -> dict:
        prompt = {
            "task": "Extract claims from the transcript and rate their stability over time return JSON {claims: [{sentenceindex:int, claimtext:str, stability:0-1}], reason: str}.",
            "transcriptsample": " ".join(sentences[:400])
        }
        resp = cachedcallaijson(client, json.dumps(prompt))
        out = {"claimpersistencescore": 1.0, "claimchangescount": 0, "claimchangesexamples": []}
        try:
            if isinstance(resp, dict) and resp.get("claims"):
                claims = resp.get("claims", [])
                stabilities = [float(c.get("stability", 1.0)) for c in claims if isinstance(c, dict)]
                out["claimpersistencescore"] = float(np.mean(stabilities)) if stabilities else 1.0
                changes = [c for c in claims if isinstance(c, dict) and c.get("stability", 1.0) < 0.5]
                out["claimchangescount"] = len(changes)
                out["claimchangesexamples"] = changes[:10]
                return out
        except Exception:
            pass
        if embarr is None:
            try:
                embarr = np.array(getembeddings(client, sentences))
            except Exception:
                embarr = None
        if embarr is None or len(sentences) < 2:
            return out
        labels = greedyclustrembeddings(embarr, threshold=0.8)
        clustrs = {}
        for i,l in enumerate(labels):
            clustrs.setdefault(l, []).append(i)
        unstable = 0
        totalclustrs = 0
        examples = []
        for cl, idxs in clustrs.items():
            if len(idxs) <= 1:
                continue
            totalclustrs += 1
            contrad = 0
            for i in range(len(idxs)):
                for j in range(i+1, len(idxs)):
                    lab, conf = classifysentencepair(client, sentences[idxs[i]], sentences[idxs[j]])
                    if lab == "contradict":
                        contrad += 1
                        examples.append({"clustr": cl, "pairs": (idxs[i], idxs[j]), "reason": "contradiction", "confidence": conf})
            if contrad > 0:
                unstable += 1
        out["claimpersistencescore"] = 1.0 - (unstable / max(1, totalclustrs)) if totalclustrs else 1.0
        out["claimchangescount"] = len(examples)
        out["claimchangesexamples"] = examples[:10]
        return out
def getembeddings(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    resp = retrybackpoff(client.embeddings.create, input=texts, model=model)
    return [d.embedding for d in resp.data]
def cosinesim(a: np.ndarray, b: np.ndarray) -> float:
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def ismathexpression(s: str) -> bool:
    """Return True if string looks like a math expression (contains digits and operators)"""
    s = s.strip()
    allowed = set("0123456789+-*/=(). ^")
    return any(ch.isdigit() for ch in s) and any(ch in allowed for ch in s)
def normalizemathequation(s: str) -> Tuple[bool, float]:
    """Try to parse an equation like '1+1=2' and return (isequation, difference)

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
        iszero = diff == 0
        return True, float(0.0 if iszero else 1.0)
    except Exception:
        return False, 0.0

AICACHEPATH = Path.home().joinpath(".monkaicache.pkl")
try:
    with open(AICACHEPATH, "rb") as fh:
        AICACHE = pickle.load(fh)
except Exception:
    AICACHE = {}
def saveaicache():
    try:
        with open(AICACHEPATH, "wb") as fh:
            pickle.dump(AICACHE, fh)
    except Exception:
        pass

def cachedcallaijson(client: OpenAI, prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.0) -> dict:
    key = hashlib.sha256((model + "|" + prompt).encode("utf-8")).hexdigest()
    if key in AICACHE:
        return AICACHE[key]
    txt = callai(client, prompt, model=model, temperature=temperature)
    try:
        out = json.loads(txt)
    except Exception:
        out = {"raw": txt}
    AICACHE[key] = out
    saveaicache()
    return out


def classifysentencepair(client: OpenAI, s1: str, s2: str) -> Tuple[str, float]:
    """Return (label, confidence) where label in {repeat, contradict, other} and confidence 0-1

    Uses math normalization and cached AI JSON classification
    """
    for s in (s1, s2):
        if ismathexpression(s):
            ok, diff = normalizemathequation(s)
            if ok and diff == 0.0:
                return "repeat", 1.0

    prompt = {
        "pair": {"a": s1, "b": s2},
        "task": "Classify whether these two phrases repeat the same idea, contradict, or neither. Return JSON {label: 'repeat'|'contradict'|'other', confidence: 0-1, reason: 'one sentence'}"
    }
    resp = cachedcallaijson(client, json.dumps(prompt))
    if not resp:
        return "other", 0.0
    label = resp.get("label") or resp.get("result") or ("repeat" if "yes" in resp.get("raw", "").lower() else "other")
    conf = float(resp.get("confidence", 0.0) or resp.get("score", 0.0) or 0.0)
    return label, conf
def syllablcount(word: str) -> int:
    w = word.lower()
    vowels = "aeiouy"
    count = 0
    prevwasvowel = False
    for ch in w:
        isvowel = ch in vowels
        if isvowel and not prevwasvowel:
            count += 1
        prevwasvowel = isvowel
    if w.endswith("e") and count > 1:
        count -= 1
    if count == 0:
        count = 1
    return count
def fleaschreadingease(text: str) -> float:
    words = [w for w in text.split() if any(c.isalpha() for c in w)]
    sentences = [s for s in text.split(".") if s.strip()]
    syllabls = sum(syllablcount(w) for w in words)
    numwords = max(1, len(words))
    numsentences = max(1, len(sentences))
    asl = numwords / numsentences
    asw = syllabls / numwords
    score = 206.835 - 1.015 * asl - 84.6 * asw
    return float(score)
def countpassiveasentences(docs: List[spacy.tokens.Doc]) -> int:
    c = 0
    for doc in docs:
        if any(tok.dep == "nsubjpass" for tok in doc):
            c += 1
    return c
def hapasxlegomenaratio(text: str) -> float:
    toks = [t.text.lower() for t in nlp(text) if t.isalpha]
    if not toks:
        return 0.0
    from collections import Counter

    cnt = Counter(toks)
    hapax = sum(1 for v in cnt.values() if v == 1)
    return hapax / len(toks)


def greedyclustrembeddings(embarr: np.ndarray, threshold: float = 0.6) -> List[int]:
    if len(embarr) == 0:
        return []
    centroids = [embarr[0].astype(float)]
    labels = [0]
    for i in range(1, len(embarr)):
        v = embarr[i].astype(float)
        sims = [cosinesim(v, c) for c in centroids]
        best = max(sims)
        if best >= threshold:
            idx = sims.index(best)
            centroids[idx] = (centroids[idx] + v) / 2.0
            labels.append(idx)
        else:
            centroids.append(v)
            labels.append(len(centroids) - 1)
    return labels
def entropyoflabels(labels: List[int]) -> float:
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
def sentencenoveltyscores(embarr: np.ndarray, window: int = 3) -> List[float]:
    n = len(embarr)
    if n == 0:
        return []
    scores = []
    for i in range(n):
        start = max(0, i - window)
        context = [embarr[j] for j in range(start, i)]
        if not context:
            scores.append(0.0)
            continue
        sims = [cosinesim(embarr[i], c) for c in context]
        meansim = float(sum(sims) / len(sims))
        scores.append(1.0 - meansim)
    return scores

def lexicaldiversity(text: str) -> float:
    toks = [t.text.lower() for t in nlp(text) if t.isalpha]
    if not toks:
        return 0.0
    return len(set(toks)) / len(toks)


def generatecandidatespans(docs: List[spacy.tokens.Doc]) -> List[dict]:
    matcher = Matcher(nlp.vocab)

    matcher.add("MODAL", [[{"TAG": "MD"}]])
    matcher.add("INTERJECTION", [[{"TAG": "UH"}], [{"POS": "INTJ"}]])
    matcher.add("PRONVERB", [[{"POS": "PRON"}, {"POS": "VERB"}]])
    matcher.add("ITVERB", [[{"LOWER": "it"}, {"POS": "VERB"}]])
    matcher.add("ADVQUAL", [[{"POS": "ADV"}], [{"DEP": "advmod"}]])

    candidates = []
    for si, doc in enumerate(docs):
        matches = matcher(doc)
        seenspans = set()
        for mid, start, end in matches:
            span = doc[start:end]
            key = (si, span.startchar, span.endchar)
            if key in seenspans:
                continue
            seenspans.add(key)
            left = max(span.start - 1, 0)
            right = min(span.end + 1, len(doc))
            extspan = doc[left:right]
            candidates.append(
                {
                    "text": extspan.text,
                    "sentenceindex": si,
                    "context": doc.text,
                    "startchar": extspan.startchar,
                    "endchar": extspan.endchar,
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


def detecthedgesfillersadvanced(client: OpenAI, text: str, sentences: List[str], docs: List[spacy.tokens.Doc]) -> dict:
    candidates = generatecandidatespans(docs)
    if not candidates:
        return {
            "hedgecount": 0,
            "fillercount": 0,
            "qualifiercount": 0,
            "defensivephrasescount": 0,
            "justificationphrasescount": 0,
            "hedgeexamples": [],
        }
    maxcands = 200
    candidates = candidates[:maxcands]
    promptpayload = {
        "candidates": [
            {"text": c["text"], "sentenceindex": c["sentenceindex"], "context": c["context"]}
            for c in candidates
        ],
        "instructions": (
            "For each candidate phrase, classify it into one of: hedge, filler, qualifier, defensive, justification, or other "
            "Also provide a numeric strength 0-1 (higher = stronger hedge/qualifier/defense) and a one sentence reason "
            "Return a JSON array of objects: {text, sentenceindex, label, strength, reason}"
        ),
    }
    prompt = (
        "You are an assistant that specializes in discourse and pragmatics "
        "Given a transcript and a list of candidate phrases that may be hedges, fillers, qualifiers, defensive phrases or justifications, classify each candidate and score its strength between 0 and 1 "
        "Respond ONLY with a JSON array as specified\n\n" + json.dumps(promptpayload)
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
            if any(tok.tag == "UH" for tok in nlp(txt)):
                label = "filler"
                strength = 0.9
            elif any(tok.tag == "MD" for tok in nlp(txt)):
                label = "hedge"
                strength = 0.8
            classified.append({"text": c["text"], "sentenceindex": c["sentenceindex"], "label": label, "strength": strength, "reason": "heuristic fallback"})
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
        "hedgecount": counts["hedge"],
        "fillercount": counts["filler"],
        "qualifiercount": counts["qualifier"],
        "defensivephrasescount": counts["defensive"],
        "justificationphrasescount": counts["justification"],
        "hedgeexamples": examples["hedge"],
        "fillerexamples": examples["filler"],
        "qualifierexamples": examples["qualifier"],
    }
def detectstorytellinganddeceptionsignals(client: OpenAI, text: str, sentences: List[str], docs: List[spacy.tokens.Doc], labels: List[int] = None, aimetrics: dict = None) -> dict:
    bridgephrases = ["so anyway", "anyway", "moving on", "as I said", "like I said", "anyways", "to be honest", "honestly", "you know", "so yeah", "but anyway", "and then"]
    lowsensorywords = set(["see","saw","seen","look","looked","seen","heard","hear","smell","smelled","taste","tasted","felt","feel","feelings","scent","sound","noise","color","colour","texture"])
    textl = text.lower()
    bridgecount = sum(textl.count(p) for p in bridgephrases)
    tokens = [t for t in nlp(text) if t.isalpha]
    totalalpha = max(1, sum(1 for t in tokens))
    sensorycount = sum(1 for t in tokens if t.lemma.lower() in lowsensorywords)
    sensoryratio = sensorycount / totalalpha
    topicclustrs = len(set(labels)) if labels is not None else 0
    sensnorm = min(1.0, sensoryratio * 5.0)
    bridgenorm = min(1.0, bridgecount / max(1, len(sentences) / 5.0))
    clustrfactor = 1.0 - min(1.0, topicclustrs / 5.0)
    scriptedvaguescore = float(min(1.0, (1.0 - sensnorm) * 0.5 + bridgenorm * 0.35 + clustrfactor * 0.15))
    adjcount = sum(1 for t in tokens if t.pos == "ADJ")
    advcount = sum(1 for t in tokens if t.pos == "ADV")
    nouncount = max(1, sum(1 for t in tokens if t.pos == "NOUN"))
    embellishmentscore = float(min(1.0, (adjcount + advcount) / nouncount / 2.0))
    wordspersentence = [len(s.split()) for s in sentences] if sentences else [0]
    avgsentencelen = float(sum(wordspersentence) / len(wordspersentence)) if wordspersentence else 0.0
    toolittledetail = 1.0 if avgsentencelen < 6.0 or (aimetrics and float(aimetrics.get("informationdensity", 0.0)) < 0.2) else 0.0
    toomuchdetail = 1.0 if avgsentencelen > 25.0 or embellishmentscore > 0.9 else 0.0
    firstperson = set(["i","me","we","us","my","our","mine","ours"])
    thirdperson = set(["he","she","they","them","his","her","their","him","hers"])
    fp = sum(1 for t in tokens if t.text.lower() in firstperson)
    tp = sum(1 for t in tokens if t.text.lower() in thirdperson)
    distancingscore = float(tp / max(1, fp + tp))
    questionsents = [s.strip() for s in sentences if s.strip().endswith("?")]
    qnorm = [q.lower() for q in questionsents]
    dupquestions = sum(1 for i,q in enumerate(qnorm) for j in range(i+1, len(qnorm)) if q == qnorm[j])
    repeatingquestioncount = dupquestions
    if len(questionsents) > 1:
        try:
            qemb = getembeddings(client, questionsents)
            qarr = np.array(qemb)
            for i in range(len(qarr)):
                for j in range(i+1, len(qarr)):
                    if cosinesim(qarr[i], qarr[j]) > 0.88:
                        repeatingquestioncount += 1
        except Exception:
            pass
    return {
        "scriptedvaguescore": scriptedvaguescore,
        "sensorydetailratio": sensoryratio,
        "bridgephrasecount": bridgecount,
        "embellishmentscore": embellishmentscore,
        "toolittledetail": toolittledetail,
        "toomuchdetail": toomuchdetail,
        "distancingscore": distancingscore,
        "repeatingquestioncount": repeatingquestioncount,
    }
def analyzetranscript(client: OpenAI, audiopath: str, outputpath: str, maxpairapicalls: int = 500):
    text, segments = loadaudiotranscript(client, audiopath)
    sentences = [s.get("text", "") for s in segments] if segments else [s.strip() for s in text.split(".") if s.strip()]
    signals = {}
    signals["sentencecount"] = len(sentences)
    docs = list(nlp.pipe(sentences))
    signals["tokencount"] = sum(len(doc) for doc in docs)
    signals["questioncount"] = sum(1 for s in sentences if s.strip().endswith("?"))
    signals["questionratio"] = signals["questioncount"] / max(1, signals["sentencecount"])
    poscounts = {"NOUN": 0, "VERB": 0, "ADJ": 0, "ADV": 0, "PRON": 0}
    stopwords = nlp.Defaults.stopwords
    funcwords = 0
    totalalpha = 0
    for tok in nlp(text):
        if tok.is_alpha:
            totalalpha += 1
            if tok.lower_ in stopwords:
                funcwords += 1
        if tok.pos_ in poscounts:
            poscounts[tok.pos_] += 1
    signals["nounratio"] = (poscounts["NOUN"] / totalalpha) if totalalpha else 0.0
    signals["verbratio"] = (poscounts["VERB"] / totalalpha) if totalalpha else 0.0
    signals["adjratio"] = (poscounts["ADJ"] / totalalpha) if totalalpha else 0.0
    signals["advratio"] = (poscounts["ADV"] / totalalpha) if totalalpha else 0.0
    signals["functionwordratio"] = (funcwords / totalalpha) if totalalpha else 0.0

    signals["exclamationcount"] = text.count("!")
    signals["commacount"] = text.count(",")
    signals["ellipsiscount"] = text.count("...")

    if segments:
        pausetimes = [segments[i+1]["start"] - segments[i]["end"] for i in range(len(segments)-1)]
        signals["pausecount"] = len([p for p in pausetimes if p > 0.7])
        signals["pausemean"] = (sum(pausetimes) / len(pausetimes)) if pausetimes else 0.0
        speechdurationsec = segments[-1]["end"] - segments[0]["start"] if len(segments) >= 2 else 0.0
    else:
        signals["pausecount"] = 0
        signals["pausemean"] = 0.0
        speechdurationsec = max(1.0, len(text.split()) / 2.0)
    durationmin = speechdurationsec / 60.0 if speechdurationsec else 1.0
    hfmetrics = detecthedgesfillersadvanced(client, text, sentences, docs)
    signals["fillersperminute"] = (hfmetrics.get("fillercount", 0) / durationmin) if durationmin else 0.0
    signals["pauseratepermin"] = (signals.get("pausecount", 0) / durationmin) if durationmin else 0.0
    def meanstrength(exlist):
        try:
            return float(np.mean([it.get("strength", 0.0) for it in exlist]))
        except Exception:
            return 0.0
    signals["hedgestrengthmean"] = meanstrength(hfmetrics.get("hedgeexamples", []))
    signals["fillerstrengthmean"] = meanstrength(hfmetrics.get("fillerexamples", []))
    from collections import Counter
    words = [t.text.lower() for t in nlp(text) if t.is_alpha]
    wcnt = Counter(words)
    topwords = wcnt.mostcommon(10)
    signals["topwordfreqs"] = topwords
    totalw = sum(wcnt.values()) if wcnt else 1
    went = 0.0
    for _k, v in wcnt.items():
        p = v / totalw
        went -= p * math.log(p + 1e-12)
    signals["wordfreqwentropya"] = float(went)
    if len(sentences) > 1 and 'labels' in locals():
        topicshifts = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])
        signals["topicshiftcount"] = int(topicshifts)
        
        from collections import Counter as C2

        csize = C2(labels)
        maxc = max(csize.values()) if csize else 0
        signals["clustrsizedisparity"] = (maxc / len(sentences)) if len(sentences) else 0.0
    else:
        signals["topicshiftcount"] = 0
        signals["clustrsizedisparity"] = 0.0

    if segments:
        pausetimes = [segments[i+1]["start"] - segments[i]["end"] for i in range(len(segments)-1)]
        signals["pausecount"] = len([p for p in pausetimes if p > 0.7])
        signals["pausemean"] = (sum(pausetimes) / len(pausetimes)) if pausetimes else 0.0
        speechdurationsec = segments[-1]["end"] - segments[0]["start"] if len(segments) >= 2 else 0.0
    else:
        signals["pausecount"] = 0
        signals["pausemean"] = 0.0
        speechdurationsec = max(1.0, len(text.split()) / 2.0)

    signals["speechratewps"] = (len(text.split()) / speechdurationsec) if speechdurationsec > 0 else 0.0

    signals["lexicaldiversity"] = lexicaldiversity(text)
    sentlens = [len(doc) for doc in docs] if docs else [0]
    signals["avgsentencelentokens"] = float(sum(sentlens) / len(sentlens)) if sentlens else 0.0
    pronouns = sum(1 for tok in nlp(text) if tok.pos == "PRON")
    tokenstotal = sum(1 for tok in nlp(text) if tok.isalpha)
    signals["pronounratio"] = (pronouns / tokenstotal) if tokenstotal else 0.0
    hfmetrics = detecthedgesfillersadvanced(client, text, sentences, docs)
    signals.update(hfmetrics)

    nestats = namedentitystats(docs)
    signals.update(nestats)
    embeddings = []
    if len(sentences) > 1:
        embeddings = getembeddings(client, sentences)
        embarr = np.array(embeddings)

        noveltyscores = sentencenoveltyscores(embarr, window=3)
        signals["noveltymean"] = float(np.mean(noveltyscores)) if noveltyscores else 0.0
        signals["noveltystd"] = float(np.std(noveltyscores)) if noveltyscores else 0.0
        signals["noveltytopcount"] = int(sum(1 for s in noveltyscores if s > 0.5))

        labels = greedyclustrembeddings(embarr, threshold=0.6)
        signals["topicclustrcount"] = int(len(set(labels)))
        signals["topicentropy"] = float(entropyoflabels(labels))

        pronounratios = []
        for doc in docs:
            pronouns = sum(1 for tok in doc if tok.pos == "PRON")
            total = sum(1 for tok in doc if tok.isalpha)
            pronounratios.append((pronouns / total) if total else 0.0)
        signals["pronounratiovariance"] = float(np.var(pronounratios)) if pronounratios else 0.0

        def dominantperson(doc):
            p1 = sum(1 for t in doc if t.lower in ("i", "me", "we", "us"))
            p2 = sum(1 for t in doc if t.lower in ("you",))
            p3 = sum(1 for t in doc if t.lower not in ("i", "me", "we", "us", "you") and t.pos == "PRON")
            if p1 >= p2 and p1 >= p3:
                return 1
            if p2 >= p1 and p2 >= p3:
                return 2
            return 3
        persons = [dominantperson(d) for d in docs]
        perspectiveshifts = sum(1 for i in range(1, len(persons)) if persons[i] != persons[i-1])
        signals["perspectiveshiftscount"] = int(perspectiveshifts)
        signals["redundancyratio"] = float(signals.get("semanticrepetitioncount", 0)) / max(1, len(sentences))
        signals["passiveavoiceratio"] = float(countpassiveasentences(docs)) / max(1, len(docs))
        clausecount = sum(sum(1 for tok in doc if tok.dep in ("ccomp", "advcl", "xcomp")) for doc in docs)
        signals["avgclausespersentence"] = float(clausecount) / max(1, len(docs))
        signals["fleaschreadingease"] = float(fleaschreadingease(text))
        signals["hapasxlegomenaratio"] = float(hapasxlegomenaratio(text))
        storysignals = detectstorytellinganddeceptionsignals(client, text, sentences, docs, labels=labels, aimetrics=aimetrics if 'aimetrics' in locals() else None)
        signals.update(storysignals)
        simthreshold = 0.6
        candidatepairs = []
        n = len(embarr)
        for i in range(n):
            for j in range(i+1, n):
                sim = cosinesim(embarr[i], embarr[j])
                if sim >= simthreshold:
                    candidatepairs.append((i, j, sim))

        forcedpairs = []
        for i in range(n):
            for j in range(i+1, n):
                s1 = sentences[i].strip()
                s2 = sentences[j].strip()
                if min(len(s1.split()), len(s2.split())) <= 4 or ismathexpression(s1) or ismathexpression(s2):
                    forcedpairs.append((i, j, 1.0))

        pairmap = {}
        for (i, j, sc) in candidatepairs + forcedpairs:
            key = (i, j)
            if key not in pairmap or sc > pairmap[key]:
                pairmap[key] = sc

        candidatepairs = [(i, j, sc) for (i, j), sc in pairmap.items()]
        candidatepairs.sort(key=lambda x: (0 if x[2] == 1.0 else 1, -x[2]))

        candidatepairs = candidatepairs[:maxpairapicalls]

        semanticrepetitioncount = 0
        contradictioncount = 0
        paraphrasereusecount = 0
        selfcorrectionscount = 0
        repeatconfidences = []
        contradictionconfidences = []

        for (i, j, sim) in tqdm(candidatepairs, desc="Sentence pair AI analysis"):
            s1 = sentences[i]
            s2 = sentences[j]
            
            label, conf = classifysentencepair(client, s1, s2)
            
            if label == "repeat":
                semanticrepetitioncount += 1
                repeatconfidences.append(conf)
            elif label == "contradict":
                contradictioncount += 1
                contradictionconfidences.append(conf)

            promptpara = f"Are these two statements paraphrases of the same idea? Answer yes or no.\n1: {s1}\n2: {s2}"
            resppara = callai(client, promptpara)
            if "yes" in resppara.lower():
                paraphrasereusecount += 1

            promptself = f"Does the second statement correct or revise the first? Answer yes or no.\n1: {s1}\n2: {s2}"
            respself = callai(client, promptself)
            if "yes" in respself.lower():
                selfcorrectionscount += 1

        signals["semanticrepetitioncount"] = semanticrepetitioncount
        signals["semanticrepetitionconfidencemean"] = float(np.mean(repeatconfidences)) if repeatconfidences else 0.0
        signals["contradictioncount"] = contradictioncount
        signals["contradictionconfidencemean"] = float(np.mean(contradictionconfidences)) if contradictionconfidences else 0.0
        signals["paraphrasereusecount"] = paraphrasereusecount
        signals["sccnt"] = selfcorrectionscount
    else:
        signals["semanticrepetitioncount"] = 0
        signals["contradictioncount"] = 0
        signals["paraphrasereusecount"] = 0
        signals["sccnt"] = 0

    topicprompt = f"Analyze the transcript and return a topic drift score between 0 and 1 (0 = stays on topic, 1 = high drift).\n\nTranscription:\n{text}"
    topicdriftscore = callai(client, topicprompt)
    try:
        topicdriftscore = float(topicdriftscore)
    except Exception:
        topicdriftscore = 0.0
    signals["tpdr"] = topicdriftscore
    aimetrics = gettranscriptoverallmetrics(client, text)
    signals["infod"] = float(aimetrics.get("informationdensity", 0.0))
    signals["certr"] = float(aimetrics.get("certaintyratio", 0.0))
    signals["politenessscore"] = float(aimetrics.get("politenessscore", 0.0))
    signals["aioverallreason"] = aimetrics.get("reason", "")
    cognutivaprompt = (
        "Analyze transcript for cognutiva load indicators: hedges, fillers, qualifiers, hesitation, uncertainty. "
        "Return a single numeric score 0-1 (higher means higher cognutiva load).\n\nTranscription:\n" + text
    )
    cognutivaloadscore = callai(client, cognutivaprompt)
    try:
        cognutivaloadscore = float(cognutivaloadscore)
    except Exception:
        cognutivaloadscore = 0.0
    signals["cogld"] = cognutivaloadscore

    emotionprompt = (
        "Analyze sentiment, emotional shifts, intensity, positive/negative variance. Return JSON with keys: "
        "sentimentmean, sentimentvariance, sentimentrange, subjectivitymean, subjectivityvariance, emotionalintensityvariance.\n\nTranscription:\n" + text
    )
    emotionmetrics = callai(client, emotionprompt)
    try:
        emotionmetrics = json.loads(emotionmetrics)
    except Exception:
        emotionmetrics = {
            "sentimentmean": 0,
            "sentimentvariance": 0,
            "sentimentrange": 0,
            "subjectivitymean": 0,
            "subjectivityvariance": 0,
            "emotionalintensityvariance": 0,
        }
    signals.update(emotionmetrics)

    def getaudioveracitymetrics(client: OpenAI, text: str, segments: List[dict], sentences: List[str], docs: List[spacy.tokens.Doc], aimetrics: dict = None, emotionmetrics: dict = None, hfmetrics: dict = None, contradictioncount: int = 0, semanticrepetitioncount: int = 0) -> dict:
        prompt = {
            "task": "Analyze the transcript for the following five signals and return a JSON object: confidence (0-1), consistency (0-1), emotionalleakage (0-1), logicalgapscount (int), logicalgapsexamples (list of {sentenceindex, reason}), logicalgapsconfidencemean (0-1), statisticalplausibility (0-1), statisticalissues (list of {text, sentenceindex, reason}), reason: short explanation.",
            "transcript": text[:4000]
        }
        resp = cachedcallaijson(client, json.dumps(prompt))
        out = {
            "cnf": 0.0,
            "cons": 0.0,
            "eml": 0.0,
            "lgcnt": 0,
            "lgex": [],
            "lgc": 0.0,
            "stpl": 1.0,
            "stiss": [],
            "avr": "",
        }
        try:
            if isinstance(resp, dict) and resp:
                out["cnf"] = float(resp.get("confidence", out["cnf"]))
                out["cons"] = float(resp.get("consistency", out["cons"]))
                out["eml"] = float(resp.get("emotionalleakage", out["eml"]))
                out["lgcnt"] = int(resp.get("logicalgapscount", out["lgcnt"]))
                out["lgex"] = resp.get("logicalgapsexamples", out["lgex"]) or []
                out["lgc"] = float(resp.get("logicalgapsconfidencemean", out["lgc"]))
                out["stpl"] = float(resp.get("statisticalplausibility", out["stpl"]))
                out["stiss"] = resp.get("statisticalissues", out["stiss"]) or []
                out["avr"] = resp.get("reason", out["avr"]) or resp.get("explanation", out["avr"]) or out["avr"]
                return out
        except Exception:
            pass
        if aimetrics and isinstance(aimetrics, dict):
            out["cnf"] = float(aimetrics.get("certaintyratio", out["cnf"]))
        else:
            hedgect = (hfmetrics.get("hedgecount", 0) if hfmetrics else 0)
            out["cnf"] = max(0.0, 1.0 - (hedgect / max(1, len(sentences))))

        contradictions = contradictioncount
        out["cons"] = max(0.0, 1.0 - (contradictions / max(1, len(sentences) / 5.0)))

        emvar = (emotionmetrics.get("emotionalintensityvariance") if emotionmetrics else 0.0)
        out["eml"] = float(min(1.0, (emvar or 0.0) * 2.0))

        conjwords = ("but", "however", "although", "though")
        gaps = [i for i,s in enumerate(sentences) if any(w in s.lower() for w in conjwords)]
        out["lgcnt"] = len(gaps)
        out["lgex"] = [{"sentenceindex": i, "reason": sentences[i][:200]} for i in gaps]
        out["lgc"] = 0.5 if gaps else 0.0
        import re
        nums = re.findall(r"\d+\.?\d*", text)
        if not nums:
            out["stpl"] = 1.0
        else:
            out["stpl"] = 0.6
            out["stiss"] = [{"text": n, "sentenceindex": next((i for i,s in enumerate(sentences) if n in s), -1), "reason": "numeric claim to verify"} for n in nums[:10]]
        out["avr"] = "heuristic fallback"
        return out
    avmetrics = getaudioveracitymetrics(client, text, segments, sentences, docs, aimetrics=aimetrics if 'aimetrics' in locals() else None, emotionmetrics=emotionmetrics, hfmetrics=hfmetrics if 'hfmetrics' in locals() else None, contradictioncount=signals.get('contradictioncount', 0), semanticrepetitioncount=signals.get('semanticrepetitioncount', 0))
    signals.update(avmetrics)

    def extracttimingandflow(segments: List[dict], sentences: List[str], text: str = "") -> dict:
        pauses = []
        latencies = []
        restarts = 0
        speechrates = []
        interruptions = []
        if segments:
            for i in range(len(segments) - 1):
                end = segments[i].get('end', 0.0)
                startnext = segments[i+1].get('start', 0.0)
                gap = max(0.0, startnext - end)
                pauses.append({'between': (i, i+1), 'duration': gap, 'time': end})
            for i, seg in enumerate(segments):
                textseg = seg.get('text', '').strip()
                duration = max(0.0001, seg.get('end', 0.0) - seg.get('start', 0.0))
                words = len([w for w in textseg.split() if w])
                speechrates.append({'segment': i, 'wps': words / duration if duration > 0 else 0.0, 'start': seg.get('start', 0.0)})
            for i,s in enumerate(sentences[:-1]):
                if s.strip().endswith('?'):
                    if i < len(segments)-1:
                        lat = segments[i+1].get('start', 0.0) - segments[i].get('end', 0.0)
                        latencies.append({'qindex': i, 'latency': lat, 'time': segments[i].get('end', 0.0)})
            for i,s in enumerate(sentences):
                parts = s.strip().split()
                if len(parts) > 1 and parts[0].lower() == parts[1].lower():
                    restarts += 1
        else:
            import re
            try:
                markers = re.findall(r"\[pause\s*([0-9]*\.?[0-9]+)s\]", text.lower()) if text else []
            except Exception:
                markers = []
            for m in markers:
                try:
                    pauses.append({'between': None, 'duration': float(m), 'time': None})
                except:
                    pass
            if pauses:
                speechrates = []
        timing = {
            'pauses': pauses,
            'latencies': latencies,
            'speechrates': speechrates,
            'restarts': restarts,
            'interruptions': interruptions,
        }
        return timing
    def analyzeprosodicvariability(acousticfeatures: dict, segments: List[dict], sentences: List[str]) -> dict:
        out = {
            'pitchspikes': [],
            'pitchflatregions': [],
            'intensityshifts': [],
            'rhythmirregularities': [],
            'emphasisconcentration': [],
            'pitchrangemean': 0.0,
            'pitchvariancemean': 0.0
        }
        if acousticfeatures and isinstance(acousticfeatures, dict):
            f0list = acousticfeatures.get('f0', [])
            intens = acousticfeatures.get('intensity', [])
            if f0list:
                meanf0 = float(np.mean(f0list))
                rng = float(np.ptp(f0list))
                var = float(np.var(f0list))
                out['pitchrangemean'] = rng
                out['pitchvariancemean'] = var
                for i,v in enumerate(f0list):
                    if v > meanf0 + 2 * (var ** 0.5):
                        out['pitchspikes'].append({'index': i, 'value': v})
                    if v < meanf0 - 2 * (var ** 0.5):
                        out['pitchflatregions'].append({'index': i, 'value': v})
            if intens:
                meani = float(np.mean(intens))
                for i,v in enumerate(intens):
                    if abs(v - meani) > 2 * (np.std(intens) if len(intens) > 1 else 0.0):
                        out['intensityshifts'].append({'index': i, 'value': v})
        else:
            markers = []
            for i,s in enumerate(sentences):
                if '[raised pitch' in s.lower() or '!' in s:
                    out['pitchspikes'].append({'segment': i, 'text': s[:120]})
                if '[lowered pitch' in s.lower():
                    out['pitchflatregions'].append({'segment': i, 'text': s[:120]})
        return out

    def analyzeadvacoust(acousticfeatures: dict, segments: List[dict], sentences: List[str], hfmetrics: dict, avmetrics: dict, signals: dict, emotionmetrics: dict, aimetrics: dict) -> dict:
        """Advanced acoustic metrics (best-effort, uses available features).
        Returns normalized 0-1 scores for many vocal markers."""
        out = {
            'vstrain': 0.0,
            'breath': 0.0,
            'vfold': 0.0,
            'artprec': 0.0,
            'vtension': 0.0,
            'microtrem': 0.0,
            'phonbreak': 0.0,
            'fry': 0.0,
            'sylrcv': 0.0,
            'artrate': 0.0,
            'fillerac': 0.0,
            'repairac': 0.0,
            'prosodycons': 0.0,
            'artcons': 0.0,
            'energy_match': 0.0,
            'timerec': 0.0,
            'prosodtext_mismatch': 0.0,
            'emotcontgap': 0.0,
            'confaccgap': 0.0,
            'effortimpmis': 0.0,
        }
        try:
            f0_all = []
            intens_all = []
            mfcc_frames = []
            formant_frames = []
            voiced_mask = []
            seg_durs = []
            for seg in segments or []:
                af = seg.get('acousticfeatures') or seg.get('acoustic') or {}
                f0 = af.get('f0')
                if isinstance(f0, list):
                    f0_all.extend([v for v in f0 if v is not None])
                elif f0 not in (None, []):
                    try:
                        f0_all.append(float(f0))
                    except:
                        pass
                I = af.get('intensity') or af.get('energy')
                if isinstance(I, list):
                    intens_all.extend([v for v in I if v is not None])
                elif I not in (None, []):
                    try:
                        intens_all.append(float(I))
                    except:
                        pass
                mf = af.get('mfcc')
                if isinstance(mf, list) and mf:
                    if isinstance(mf[0], (list, tuple, np.ndarray)):
                        mfcc_frames.extend(mf)
                    else:
                        mfcc_frames.append(mf)
                frm = af.get('formants')
                if isinstance(frm, list) and frm:
                    formant_frames.extend([f for f in frm if f])
                vm = af.get('voiced')
                if isinstance(vm, list):
                    voiced_mask.extend(vm)
                seg_durs.append(max(0.0001, float(seg.get('end', 0.0) - seg.get('start', 0.0))))

            # Vocal Strain
            mfcc_var = 0.0
            if mfcc_frames:
                arr = np.array(mfcc_frames, dtype=float)
                var_by_coef = np.var(arr, axis=0)
                mfcc_var = float(np.mean(var_by_coef))
            shimmer = 0.0
            try:
                if intens_all:
                    amp = np.array(intens_all, dtype=float)
                    shimmer = float(np.mean(np.abs(np.diff(amp)))) / (float(np.mean(amp)) + 1e-12)
            except:
                shimmer = 0.0
            out['vstrain'] = min(1.0, (mfcc_var * 0.01) + min(1.0, shimmer))

            # Breath control: pause-to-speech
            gaps = []
            for i in range(len(segments or []) - 1):
                e = segments[i].get('end', 0.0)
                s2 = segments[i+1].get('start', 0.0)
                gaps.append(max(0.0, s2 - e))
            pause_total = sum([g for g in gaps if g > 0.3])
            speech_total = sum(seg_durs) if seg_durs else 1.0
            out['breath'] = min(1.0, pause_total / (speech_total + 1e-12))

            # Vocal fold stability: jitter & HNR
            jitter = 0.0
            hnr = 0.0
            try:
                if f0_all:
                    f0a = np.array(f0_all)
                    dif = np.diff(f0a)
                    jitter = float(np.mean(np.abs(dif))) / (np.mean(f0a) + 1e-12)
                hnrs = [ (seg.get('acousticfeatures') or seg.get('acoustic') or {}).get('hnr') for seg in segments or []]
                hnrs = [h for h in hnrs if isinstance(h, (int, float))]
                if hnrs:
                    hnr = float(np.mean(hnrs))
            except:
                jitter = 0.0
                hnr = 0.0
            out['vfold'] = float(max(0.0, min(1.0, (1.0 - min(1.0, jitter * 10.0)) * (min(1.0, (hnr / 30.0))))))

            # Articulation precision via formants
            fc = 0.0
            try:
                if formant_frames:
                    ff = np.array(formant_frames, dtype=float)
                    if ff.ndim == 2 and ff.shape[1] >= 2:
                        df = ff[:,1] - ff[:,0]
                        fc = float(np.mean(df) / (np.mean(ff[:,0]) + 1e-12))
            except:
                fc = 0.0
            out['artprec'] = float(min(1.0, max(0.0, fc * 0.01)))

            # Vocal tension: spectral centroid shift
            centroids = []
            try:
                for seg in segments or []:
                    cent = (seg.get('acousticfeatures') or seg.get('acoustic') or {}).get('spectral_centroid')
                    if isinstance(cent, (int, float)):
                        centroids.append(float(cent))
                if centroids:
                    base = float(np.mean(centroids[:max(1, len(centroids)//4)]))
                    shift = float(np.mean(centroids) - base)
                    out['vtension'] = float(min(1.0, max(0.0, shift / (base + 1e-12))))
            except:
                out['vtension'] = 0.0

            # Micro-tremors: f0 mod < 20Hz
            mt = 0.0
            try:
                if f0_all and len(f0_all) > 8:
                    f0a = np.array(f0_all, dtype=float)
                    f0d = f0a - np.mean(f0a)
                    ps = np.abs(np.fft.rfft(f0d))**2
                    freqs = np.fft.rfftfreq(len(f0d), d=0.01)
                    low = ps[(freqs>0) & (freqs < 20.0)]
                    mt = float(np.sum(low) / (np.sum(ps) + 1e-12))
            except:
                mt = 0.0
            out['microtrem'] = float(min(1.0, mt * 10.0))

            # Phonation breaks
            pb = 0.0
            try:
                if voiced_mask:
                    vm = np.array([1 if v else 0 for v in voiced_mask])
                    zeros = np.sum((vm==0) & (np.array(intens_all) > (np.mean(intens_all) if intens_all else 0))) if intens_all else 0
                    pb = float(min(1.0, zeros / (len(vm)+1e-12)))
            except:
                pb = 0.0
            out['phonbreak'] = pb

            # Vocal fry: low band energy
            fry = 0.0
            try:
                lows = []
                highs = []
                for seg in segments or []:
                    be = (seg.get('acousticfeatures') or seg.get('acoustic') or {}).get('band_energy')
                    if isinstance(be, dict):
                        l = be.get('0-120') or be.get('0_120')
                        h = sum(v for k,v in be.items() if k not in ('0-120','0_120')) if isinstance(be, dict) else 0
                        if l is not None:
                            lows.append(float(l))
                        if h is not None:
                            highs.append(float(h))
                if lows:
                    fry = float(np.mean(lows) / (np.mean(lows + highs) + 1e-12))
            except:
                fry = 0.0
            out['fry'] = float(min(1.0, fry * 2.0))

            # Syllable rate variability and articulation rate
            try:
                syll_rates = []
                for seg in segments or []:
                    txt = seg.get('text','').strip()
                    dur = max(0.0001, seg.get('end',0.0)-seg.get('start',0.0))
                    syls = sum(syllablecount(w) for w in txt.split()) if txt else 0
                    syll_rates.append(syls / dur if dur>0 else 0.0)
                if syll_rates:
                    sylrcv = float(np.std(syll_rates) / (np.mean(syll_rates) + 1e-12))
                    out['sylrcv'] = float(min(1.0, sylrcv))
                    tot_speech = sum(seg_durs)
                    tot_pause = sum([g for g in gaps if g > 0.05])
                    out['artrate'] = float(min(1.0, (sum([len(s.split()) for s in sentences]) / (tot_speech - tot_pause + 1e-12)) / 5.0))
            except:
                out['sylrcv'] = 0.0
                out['artrate'] = 0.0

            # Filler acoustics
            try:
                filler_feats = []
                content_feats = []
                for seg in segments or []:
                    txt = seg.get('text','').lower()
                    af = seg.get('acousticfeatures') or seg.get('acoustic') or {}
                    vec = af.get('mfcc') or af.get('f0') or af.get('intensity')
                    if not vec:
                        continue
                    val = float(np.mean(np.array(vec).astype(float))) if isinstance(vec,(list,tuple,np.ndarray)) else float(vec)
                    if any(x in txt for x in ('um','uh','mm')):
                        filler_feats.append(val)
                    else:
                        content_feats.append(val)
                if filler_feats and content_feats:
                    d = abs(np.mean(filler_feats) - np.mean(content_feats))
                    out['fillerac'] = float(min(1.0, d / (abs(np.mean(content_feats)) + 1e-12)))
            except:
                out['fillerac'] = 0.0

            # Repair acoustics
            try:
                repair_feats_pre = []
                repair_feats_post = []
                for i, seg in enumerate(segments or []):
                    txt = seg.get('text','').lower()
                    if 'i mean' in txt or 'sorry' in txt or 'i mean to say' in txt:
                        if i>0:
                            afp = segments[i-1].get('acousticfeatures') or segments[i-1].get('acoustic') or {}
                            vpre = afp.get('mfcc') or afp.get('intensity') or afp.get('f0')
                            if vpre:
                                repair_feats_pre.append(np.mean(np.array(vpre).astype(float)))
                        afn = seg.get('acousticfeatures') or seg.get('acoustic') or {}
                        vpost = afn.get('mfcc') or afn.get('intensity') or afn.get('f0')
                        if vpost:
                            repair_feats_post.append(np.mean(np.array(vpost).astype(float)))
                if repair_feats_pre and repair_feats_post:
                    out['repairac'] = float(min(1.0, abs(np.mean(repair_feats_post) - np.mean(repair_feats_pre)) / (abs(np.mean(repair_feats_pre)) + 1e-12)))
            except:
                out['repairac'] = 0.0

            # Consistency measures
            try:
                claims = []
                # best-effort claim extraction
                try:
                    resp = cachedcallaijson(analyzeadvacoust.__globals__.get('initclient', lambda x=None: None)(), json.dumps({"task":"extract claims","transcriptsample":" ".join(sentences[:200])}))
                    if isinstance(resp, dict) and resp.get('claims'):
                        claims = resp.get('claims', [])
                except Exception:
                    claims = []
                corrs = []
                artcorrs = []
                for c in claims:
                    if not isinstance(c, dict):
                        continue
                    txt = c.get('claimtext','').lower()
                    if not txt:
                        continue
                    segidxs = [i for i,s in enumerate(segments or []) if txt in (s.get('text','').lower())]
                    if len(segidxs) > 1:
                        mats = []
                        mats_f = []
                        for idx in segidxs:
                            af = segments[idx].get('acousticfeatures') or segments[idx].get('acoustic') or {}
                            f = af.get('f0') or []
                            mf = af.get('formants') or []
                            if f:
                                mats.append(np.array(f, dtype=float))
                            if mf:
                                try:
                                    mats_f.append(np.mean(np.array([np.mean(fr) for fr in mf])))
                                except:
                                    pass
                        for i1 in range(len(mats)):
                            for i2 in range(i1+1, len(mats)):
                                a = mats[i1]
                                b = mats[i2]
                                L = min(len(a), len(b))
                                if L > 3:
                                    a2 = (a[:L]-np.mean(a[:L]))
                                    b2 = (b[:L]-np.mean(b[:L]))
                                    den = (np.std(a2)+1e-12)*(np.std(b2)+1e-12)
                                    corr = float(np.mean((a2/ (np.std(a2)+1e-12)) * (b2/ (np.std(b2)+1e-12))))
                                    corrs.append(corr)
                        if mats_f and len(mats_f)>1:
                            artcorrs.append(1.0 - (np.std(mats_f) / (np.mean(mats_f)+1e-12)))
                out['prosodycons'] = float(min(1.0, np.mean(corrs) if corrs else 0.0))
                out['artcons'] = float(min(1.0, np.mean(artcorrs) if artcorrs else 0.0))
            except:
                out['prosodycons'] = 0.0
                out['artcons'] = 0.0

            # Energy & timing recurrence
            try:
                energy_corrs = []
                timerec = []
                for i in range(len(segments or [])):
                    for j in range(i+1, len(segments or [])):
                        s1 = segments[i].get('acousticfeatures') or segments[i].get('acoustic') or {}
                        s2 = segments[j].get('acousticfeatures') or segments[j].get('acoustic') or {}
                        e1 = s1.get('intensity') or []
                        e2 = s2.get('intensity') or []
                        if e1 and e2:
                            L = min(len(e1), len(e2))
                            if L>3:
                                a = np.array(e1[:L]) - np.mean(e1[:L])
                                b = np.array(e2[:L]) - np.mean(e2[:L])
                                energy_corrs.append(float(np.mean((a/ (np.std(a)+1e-12))*(b/ (np.std(b)+1e-12)))))
                        g1 = (segments[i+1].get('start',0)-segments[i].get('end',0)) if i+1 < len(segments) else 0
                        g2 = (segments[j+1].get('start',0)-segments[j].get('end',0)) if j+1 < len(segments) else 0
                        if abs(g1-g2) < 0.05:
                            timerec.append(1)
                out['energy_match'] = float(min(1.0, np.mean(energy_corrs) if energy_corrs else 0.0))
                out['timerec'] = float(min(1.0, sum(timerec)/(len(timerec)+1e-12)))
            except:
                out['energy_match'] = 0.0
                out['timerec'] = 0.0

            # Prosodic-textual mismatch and gaps
            try:
                cert = float(aimetrics.get('certaintyratio', 0.0) if isinstance(aimetrics, dict) else 0.0)
                hes = (hfmetrics.get('fillerdensity',0.0) if hfmetrics else 0.0) + (out.get('breath',0.0))
                out['prosodtext_mismatch'] = float(min(1.0, max(0.0, cert * hes)))
                sent_mean = float(emotionmetrics.get('sentimentmean', 0.0) if isinstance(emotionmetrics, dict) else 0.0)
                tense = out.get('vtension',0.0) + (1.0 - out.get('vfold',0.0))
                out['emotcontgap'] = float(min(1.0, max(0.0, (1.0 - abs(sent_mean)) * (tense))))
                vunc = 1.0 - out.get('vfold',0.0)
                avcnf = float(avmetrics.get('cnf', 0.0) if isinstance(avmetrics, dict) else 0.0)
                out['confaccgap'] = float(min(1.0, max(0.0, avcnf * vunc)))
                ent = float(np.mean(intens_all) if intens_all else 0.0)
                infod = float(aimetrics.get('informationdensity',0.0) if isinstance(aimetrics, dict) else 0.0)
                out['effortimpmis'] = float(min(1.0, max(0.0, (ent/ (np.mean(intens_all)+1e-12 if intens_all else 1.0)) * (1.0 - infod))))
            except:
                out['prosodtext_mismatch'] = 0.0
                out['emotcontgap'] = 0.0
                out['confaccgap'] = 0.0
                out['effortimpmis'] = 0.0
        except Exception:
            return out
        return out

    def hesitationandrepairsignals(sentences: List[str], docs: List[spacy.tokens.Doc], hfmetrics: dict) -> dict:
        out = {
            'fillerdensity': 0.0,
            'selfcorrections': 0,
            'abortedphrases': 0,
            'repeatedwords': 0,
            'elongations': 0,
            'clustrs': []
        }
        fillerct = hfmetrics.get('fillercount', 0) if hfmetrics else 0
        out['fillerdensity'] = fillerct / max(1, len(sentences))
        out['selfcorrections'] = int(signals.get('sccnt', 0))
        import re
        for i,s in enumerate(sentences):
            if '--' in s or '...' in s:
                out['abortedphrases'] += 1
            if re.search(r"\b(\w+)\s+\1\b", s.lower()):
                out['repeatedwords'] += 1
            if re.search(r"([a-zA-Z])\1{3,}", s):
                out['elongations'] += 1
            if out['abortedphrases'] or out['repeatedwords'] or out['elongations']:
                out['clustrs'].append({'index': i, 's': s[:120]})
        return out
    def stabovertime(metricssequence: List[float], window: int = 3) -> dict:
        if not metricssequence:
            return {'stabilityscore': 1.0, 'changes': []}
        changes = []
        import statistics
        for i in range(len(metricssequence) - window):
            w = metricssequence[i:i+window]
            try:
                if statistics.pstdev(w) > 0.2:
                    changes.append({'index': i, 'windowstd': statistics.pstdev(w)})
            except:
                pass
        stabilityscore = 1.0 - (len(changes) / max(1, len(metricssequence)))
        return {'stabilityscore': float(max(0.0, min(1.0, stabilityscore))), 'changes': changes}
    def stabilityovertime(metricssequence: List[float], window: int = 9999) -> dict:
        return stabovertime(metricssequence, window=window)
    def crossmodalconsistencycheck(aicertainty: float, vocalconfidence: float) -> dict:
        mismatch = abs(aicertainty - vocalconfidence)
        return {'consistencymismatch': mismatch, 'flag': mismatch > 0.4}
    timing = extracttimingandflow(segments, sentences, text)
    acousticfeatures = None
    if segments:
        f0list = []
        intensitylist = []
        for seg in segments:
            af = seg.get('acousticfeatures') or seg.get('acoustic') or {}
            if isinstance(af, dict):
                if 'f0' in af:
                    v = af.get('f0')
                    if isinstance(v, list):
                        f0list.extend(v)
                    elif v is not None:
                        f0list.append(v)
                if 'intensity' in af:
                    v = af.get('intensity')
                    if isinstance(v, list):
                        intensitylist.extend(v)
                    elif v is not None:
                        intensitylist.append(v)
        if f0list or intensitylist:
            acousticfeatures = {'f0': f0list, 'intensity': intensitylist}
    prosodic = analyzeprosodicvariability(acousticfeatures, segments, sentences)
    sents = sentences if 'sentences' in locals() else []
    adv = analyzeadvacoust(acousticfeatures, segments, sents, hfmetrics, avmetrics, signals if 'signals' in locals() else {}, emotionmetrics if 'emotionmetrics' in locals() else {}, aimetrics if 'aimetrics' in locals() else {})
    try:
        signals.update(adv)
    except Exception:
        pass
    prodi = prosodic
    hesitation = hesitationandrepairsignals(sentences, docs, hfmetrics)
    tm = timing
    control = computevocalcontrollevel(prodi, tm)
    ctrl = control
    ctrllvl = ctrl
    speechrateseq = [x.get('wps', 0.0) for x in tm.get('speechrates', [])]
    stability = stabilityovertime(speechrateseq)
    aicert = float(signals.get('certr', 0.0))
    vocalconf = float(avmetrics.get('cnf', 0.0))
    cross = crossmodalconsistencycheck(aicert, vocalconf)
    vocalsignalmap = {
        'Delivery stability': ctrllvl,
        'cognutiva load indicators': 'Medium' if signals.get('cogld', 0.0) > 0.4 else 'Low',
        'Emotional pressure indicators': 'Localized' if signals.get('emotionalintensityvariance', 0.0) > 0.2 else 'None',
        'Vocal control level': ctrllvl,
        'Timing irregularities': 'Widespread' if any(p['duration'] > 1.0 for p in timing.get('pauses', [])) else ('Localized' if any(p['duration'] > 0.5 for p in timing.get('pauses', [])) else 'None'),
        'Hesitation density': 'High' if hesitation.get('fillerdensity', 0.0) > 0.08 or hesitation.get('repeatedwords', 0) > 2 else ('Medium' if hesitation.get('fillerdensity', 0.0) > 0.03 else 'Low')
    }
    timestampedsignals = []
    for p in timing.get('pauses', []):
        if p.get('duration', 0.0) > 0.5:
            timestampedsignals.append({ 'time': p.get('time'), 'signal': f"pause {p.get('duration'):.2f}s" })
    for l in timing.get('latencies', []):
        if l.get('latency', 0.0) > 1.0:
            timestampedsignals.append({ 'time': l.get('time'), 'signal': f"latency {l.get('latency'):.2f}s" })
    for h in hesitation.get('clustrs', []):
        timestampedsignals.append({ 'time': None, 'signal': f"hesitation clustr at sentence {h.get('index')}" })

    pressurepoints = [s.get('time') for s in timing.get('pauses', []) if s.get('duration', 0.0) > 1.0]

    observationalnotes = [
        'Timing analysis from segments and textual markers',
        'Prosodic analysis uses acoustic features when available; otherwise uses textual markers'
    ]

    limitations = []
    if not signals.get('acousticfeatures'):
        limitations.append('Acoustic features missing; prosodic measures inferred from textual markers and less reliable')

    def _qual(v, rev=False):
        try:
            v = float(v)
        except Exception:
            return 'Unknown'
        if rev:
            return 'High' if v > 0.6 else ('Medium' if v > 0.3 else 'Low')
        else:
            return 'High' if v > 0.6 else ('Medium' if v > 0.3 else 'Low')

    acousticsignalmap = {
        'Pitch stability': 'Low' if prosodic.get('pitchvariancemean', 0.0) > 50 else ('Medium' if prosodic.get('pitchvariancemean', 0.0) > 20 else 'High'),
        'Intensity stability': 'Low' if len(prosodic.get('intensityshifts', [])) > 3 else ('Medium' if len(prosodic.get('intensityshifts', [])) > 0 else 'High'),
        'Spectral stability': 'Low' if len(signals.get('stiss', [])) > 0 else 'High',
        'Temporal regularity': 'Low' if len([p for p in timing.get('pauses', []) if p.get('duration', 0.0) > 0.5]) > 3 else 'High',
        'Voice noise indicators': 'High' if signals.get('avr') == 'heuristic fallback' and len(signals.get('stiss', []))>0 else 'Low',
        'Overall acoustic variability': 'High' if prosodic.get('pitchvariancemean', 0.0) > 40 or len(prosodic.get('intensityshifts', []))>2 else 'Medium',
        'Vocal strain': _qual(signals.get('vstrain', 0.0)),
        'Breath control': _qual(signals.get('breath', 0.0)),
        'Vocal fold stability': _qual(signals.get('vfold', 0.0), rev=True),
        'Articulation precision': _qual(signals.get('artprec', 0.0), rev=True),
        'Vocal tension': _qual(signals.get('vtension', 0.0)),
        'Micro-tremors': _qual(signals.get('microtrem', 0.0)),
        'Phonation breaks': _qual(signals.get('phonbreak', 0.0)),
        'Vocal fry': _qual(signals.get('fry', 0.0)),
        'Syllable rate var': _qual(signals.get('sylrcv', 0.0)),
        'Articulation rate': _qual(signals.get('artrate', 0.0)),
        'Filler acoustics diff': _qual(signals.get('fillerac', 0.0)),
        'Repair acoustics diff': _qual(signals.get('repairac', 0.0)),
        'Prosodic consistency': _qual(signals.get('prosodycons', 0.0), rev=True),
        'Articulatory consistency': _qual(signals.get('artcons', 0.0), rev=True),
        'Energy contour match': _qual(signals.get('energy_match', 0.0), rev=True),
        'Timing pattern recurrence': _qual(signals.get('timerec', 0.0), rev=True),
        'Prosodic-text mismatch': _qual(signals.get('prosodtext_mismatch', 0.0)),
        'Emotional-content gap': _qual(signals.get('emotcontgap', 0.0)),
        'Confidence-Accuracy gap': _qual(signals.get('confaccgap', 0.0)),
        'Effort-Importance mismatch': _qual(signals.get('effortimpmis', 0.0)),
    }

    signals['vocalsignalmap'] = vocalsignalmap
    signals['timestampedvocalevents'] = timestampedsignals
    signals['pressurepoints'] = pressurepoints
    signals['observationalnotes'] = observationalnotes
    signals['limitations'] = limitations
    signals['acousticsignalmap'] = acousticsignalmap
    signals['crossmodalmismatch'] = cross
    pass
    redefmetrics = detectredefinitionsandscopeshifts(client, text, sentences)
    signals.update(redefmetrics)
    claimmetrics = getclaimstability(client, sentences, embarr if 'embarr' in locals() else None)
    signals.update(claimmetrics)
    signals["ambig"] = float(min(1.0, (hfmetrics.get("hedgecount", 0) + hfmetrics.get("qualifiercount", 0)) / max(1, len(sentences)) ))
    signals["hqr"] = float((hfmetrics.get("hedgecount", 0) / max(1, hfmetrics.get("qualifiercount", 1))))

    if len(sentences) > 1000000:
        try:
            print("huge transcript, bail maybe")
        except:
            pass

    outputpath = Path(outputpath)
    outputpath.parent.mkdir(parents=True, exist_ok=True)
    def _to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)
    with open(outputpath, "w", encoding="utf-8") as fh:
        json.dump(signals, fh, indent=2, default=_to_serializable)
    print(f"Took out {len(signals)} signals and saved to {outputpath}")
    return signals

    def computevocalcontrollevel(*args, **kwargs):
        try:
            return computeVocalControl(*args, **kwargs)
        except Exception:
            return 'Low'
def parse_args():
    p = argparse.ArgumentParser(description="EchoScore transcript signal extractor")
    p.add_argument("--audio", type=str, default="harvard.wav", help="path to audio file to transcribe")
    p.add_argument("--out", type=str, default="monksignals.json", help="output JSON path")
    p.add_argument("--api-key", type=str, default=None, help="OpenAI API key; if not set, uses OPENAI_API_KEY env var")
    p.add_argument("--max-pair-api-calls", dest="max_pair_api_calls", type=int, default=500, help="max sentence pair API calls via embeddings prefilter")
    return p.parse_args()
if __name__ == "__main__":
    args = parse_args()
    client = initclient(args.api_key)
    analyzetranscript(client, args.audio, args.out, maxpairapicalls=args.max_pair_api_calls)
