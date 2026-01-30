#!/usr/bin/env python3
"""Experiment 2: Embedding-based incoming message classification.

Instead of using LLM to classify what response is needed,
use embeddings to classify the INCOMING message type, then
map to expected response type.

Flow:
1. Classify incoming message type (embedding similarity)
2. Map to expected response type (simple rules)
3. Generate with that constraint

Usage:
    python scripts/exp2_embedding_classifier.py --samples 50
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

CLEAN_TEST_SET = Path("results/test_set/clean_test_data.jsonl")
RESULTS_DIR = Path("results/experiments")


# =============================================================================
# INCOMING MESSAGE CLASSIFIER (embedding-based)
# =============================================================================

class IncomingClassifier:
    """Classifies what TYPE of message the incoming message is."""

    # Anchors for INCOMING message types (what they sent us)
    INCOMING_ANCHORS = {
        "invitation": [
            "wanna hang?", "want to hang out?", "can you come?", "are you free?",
            "want to join?", "you down?", "coming tonight?", "want to come over?",
            "are you coming?", "can you make it?", "want to grab food?",
            "let's hang", "let's meet up", "we should hang",
        ],
        "question": [
            "what time?", "where is it?", "when?", "who's coming?", "who's going?",
            "what's the address?", "what should I bring?", "how do I get there?",
            "what's the plan?", "when does it start?", "where should we meet?",
        ],
        "news": [
            "guess what", "you won't believe this", "check this out", "omg",
            "i just got promoted", "i got the job", "we broke up", "i passed",
            "dude", "bro", "yo listen", "so this happened",
        ],
        "reaction": [
            "lol", "haha", "that's hilarious", "no way", "wild", "crazy",
            "damn", "wow", "nice", "sick", "fr", "deadass", "bruh",
        ],
        "logistics": [
            "on my way", "running late", "i'll be there at", "just left",
            "almost there", "be there in 5", "leaving now", "eta 10 min",
            "i'm here", "waiting outside", "parked",
        ],
        "info_share": [
            "it's at 7", "the address is", "meet at", "starts at",
            "i'll bring", "i'm bringing", "it's tomorrow", "it's on saturday",
        ],
        "thanks": [
            "thanks", "thank you", "appreciate it", "thanks so much",
            "thx", "ty", "thanks man", "thanks bro",
        ],
        "statement": [
            "i can't make it", "i'll be late", "sorry", "my bad",
            "i forgot", "i'm busy", "something came up",
        ],
    }

    # Map incoming type → expected response type
    RESPONSE_MAP = {
        "invitation": ["accept", "decline"],  # They invited us
        "question": ["info"],                  # They asked, we answer
        "news": ["reaction"],                  # They shared news, we react
        "reaction": ["reaction", "acknowledge"],  # They reacted, we react back
        "logistics": ["acknowledge"],          # They gave update, we ack
        "info_share": ["acknowledge"],         # They gave info, we ack
        "thanks": ["acknowledge"],             # They thanked, we ack
        "statement": ["acknowledge", "reaction"],  # They stated something
    }

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Pre-compute anchor embeddings
        self.anchor_embeddings = {}
        for msg_type, phrases in self.INCOMING_ANCHORS.items():
            self.anchor_embeddings[msg_type] = self.model.encode(
                phrases, normalize_embeddings=True
            )

    def classify_incoming(self, message: str) -> tuple[str, float]:
        """Classify what type of message this is."""
        if not message.strip():
            return "unknown", 0.0

        msg_emb = self.model.encode([message], normalize_embeddings=True)[0]

        best_type, best_score = "unknown", -1
        for msg_type, anchor_embs in self.anchor_embeddings.items():
            max_sim = float(np.max(np.dot(anchor_embs, msg_emb)))
            if max_sim > best_score:
                best_score = max_sim
                best_type = msg_type

        return best_type, best_score

    def get_expected_response(self, incoming_type: str) -> list[str]:
        """Map incoming message type to expected response types."""
        return self.RESPONSE_MAP.get(incoming_type, ["acknowledge"])


# =============================================================================
# RESPONSE CLASSIFIER (for evaluating generated responses)
# =============================================================================

class ResponseClassifier:
    """Classifies what intent a RESPONSE has."""

    RESPONSE_ANCHORS = {
        "accept": [
            "yes", "yeah", "sure", "sounds good", "down", "i'm in", "definitely",
            "yep", "let's do it", "i'm down", "for sure", "bet", "count me in",
        ],
        "decline": [
            "no", "nah", "can't make it", "sorry can't", "busy", "not going",
            "pass", "maybe later", "i can't", "won't be able to",
        ],
        "question": [
            "what time", "when is it", "where at", "why", "how", "what's the plan",
            "which one", "who's coming", "what happened", "where",
        ],
        "reaction": [
            "lol", "haha", "nice", "wow", "crazy", "damn", "omg", "that's funny",
            "hilarious", "no way", "wild", "bruh", "fr", "congrats", "ayy",
        ],
        "info": [
            "i'll be there at", "at 5pm", "tomorrow morning", "the address is",
            "it's at", "i'm at", "heading there now", "running late", "omw",
        ],
        "acknowledge": [
            "ok", "got it", "understood", "cool", "bet", "alright", "noted",
            "aight", "k", "np", "sounds good", "perfect", "works for me",
        ],
    }

    def __init__(self, model):
        self.model = model
        self.anchor_embeddings = {}
        for intent, phrases in self.RESPONSE_ANCHORS.items():
            self.anchor_embeddings[intent] = self.model.encode(
                phrases, normalize_embeddings=True
            )

    def classify(self, message: str) -> tuple[str, float]:
        """Classify the intent of a response."""
        if not message.strip():
            return "unknown", 0.0

        msg_emb = self.model.encode([message], normalize_embeddings=True)[0]

        best_intent, best_score = "unknown", -1
        for intent, anchor_embs in self.anchor_embeddings.items():
            max_sim = float(np.max(np.dot(anchor_embs, msg_emb)))
            if max_sim > best_score:
                best_score = max_sim
                best_intent = intent

        return best_intent, best_score


# =============================================================================
# GENERATION PROMPTS BY RESPONSE TYPE
# =============================================================================

RESPONSE_PROMPTS = {
    "accept": """Reply accepting. Casual texting, 1-5 words.

them: wanna hang? → yeah down
them: coming tonight? → yea for sure
them: can you make it? → yeah i'll be there

them: {last_message}
me:""",

    "decline": """Reply declining. Casual texting, 2-6 words.

them: wanna hang? → nah can't today
them: coming tonight? → sorry can't make it
them: can you make it? → nah i'm busy

them: {last_message}
me:""",

    "info": """Give info they asked for. Casual texting, 2-8 words.

them: what time? → like 7ish
them: where is it? → at mike's place
them: when should I come? → anytime after 6

them: {last_message}
me:""",

    "reaction": """React to their message. Casual texting, 1-4 words.

them: guess what → what happened
them: i got the job → ayy congrats
them: that was wild → fr tho

them: {last_message}
me:""",

    "acknowledge": """Acknowledge their message. Casual texting, 1-3 words.

them: i'll be there at 5 → bet
them: on my way → cool
them: thanks for helping → np

them: {last_message}
me:""",

    "question": """Ask a follow-up question. Casual texting, 2-5 words.

them: let's meet up → when works for u
them: i have an idea → what is it
them: we should do something → like what

them: {last_message}
me:""",
}


def clean_response(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    for prefix in ["me:", "Me:", "REPLY:", "Reply:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = text.split("\n")[0].strip()
    return text


def load_test_set(limit: int) -> list[dict]:
    samples = []
    with open(CLEAN_TEST_SET) as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= limit:
                break
    return samples


@dataclass
class ExpResult:
    sample_id: int
    last_message: str
    incoming_type: str
    incoming_score: float
    expected_responses: list[str]
    gold: str
    gold_intent: str
    generated: str
    gen_intent: str
    intent_match: bool
    expected_match: bool  # Does gold intent match what we expected?


def run_experiment(model_name: str, n_samples: int, verbose: bool):
    from core.models.loader import ModelLoader

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT 2: Embedding-Based Incoming Classification")
    print(f"  Model: {model_name}")
    print(f"  Samples: {n_samples}")
    print(f"{'='*70}")

    # Load components
    print("\nLoading test set...")
    samples = load_test_set(n_samples)

    print("Loading incoming classifier...")
    incoming_clf = IncomingClassifier()

    print("Loading response classifier...")
    response_clf = ResponseClassifier(incoming_clf.model)  # Share the model

    print(f"Loading LLM ({model_name})...")
    loader = ModelLoader(model_name)
    loader.preload()

    # Run experiment
    results = []
    stop_seqs = ["\n", "them:", "<|im_end|>", "<|eot_id|>"]

    print(f"\nRunning embedding-based classification + generation...")
    start_time = time.time()

    intent_matches = 0
    expected_matches = 0

    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_samples}] Intent match: {intent_matches}/{i} ({intent_matches/(i or 1)*100:.0f}%)")

        last_msg = sample.get("last_message", "")
        gold = sample["gold_response"]

        # Step 1: Classify incoming message type (embedding-based)
        incoming_type, incoming_score = incoming_clf.classify_incoming(last_msg)
        expected_responses = incoming_clf.get_expected_response(incoming_type)

        # Classify gold response
        gold_intent, _ = response_clf.classify(gold)

        # Check if our expectation matches reality
        exp_match = gold_intent in expected_responses
        if exp_match:
            expected_matches += 1

        # Step 2: Generate with the expected response type
        # Pick first expected response type (could also try multiple)
        response_type = expected_responses[0]

        # Handle accept/decline - for now just try accept
        if response_type in ["accept", "decline"]:
            # Could be either - just pick one for now
            # In real usage, we'd ask the user or use context
            response_type = "accept"  # Default to accepting

        gen_prompt = RESPONSE_PROMPTS.get(response_type, RESPONSE_PROMPTS["acknowledge"])
        gen_prompt = gen_prompt.format(last_message=last_msg)

        gen_result = loader.generate(
            prompt=gen_prompt,
            max_tokens=20,
            temperature=0.3,
            stop=stop_seqs,
        )
        generated = clean_response(gen_result.text)
        gen_intent, _ = response_clf.classify(generated)

        # Check intent match
        match = (gold_intent == gen_intent)
        if match:
            intent_matches += 1

        result = ExpResult(
            sample_id=i,
            last_message=last_msg,
            incoming_type=incoming_type,
            incoming_score=incoming_score,
            expected_responses=expected_responses,
            gold=gold,
            gold_intent=gold_intent,
            generated=generated,
            gen_intent=gen_intent,
            intent_match=match,
            expected_match=exp_match,
        )
        results.append(result)

        if verbose:
            marker = "✓" if match else "✗"
            exp_marker = "✓" if exp_match else "✗"
            print(f"    {marker} [{incoming_type}→{expected_responses}] gold={gold_intent}{exp_marker} gen={gen_intent}")
            print(f"       \"{last_msg[:30]}\" → \"{generated[:30]}\"")

    elapsed = time.time() - start_time

    # Print results
    n = len(results)
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Time: {elapsed:.1f}s ({elapsed/n:.2f}s/sample)")

    print(f"\n--- METRICS ---")
    print(f"  Intent Match:           {intent_matches}/{n} ({intent_matches/n*100:.1f}%)")
    print(f"  Expectation Accuracy:   {expected_matches}/{n} ({expected_matches/n*100:.1f}%)")
    print(f"    (Does our classification predict the right response type?)")

    # By incoming type
    print(f"\n--- BY INCOMING MESSAGE TYPE ---")
    for msg_type in incoming_clf.INCOMING_ANCHORS.keys():
        type_results = [r for r in results if r.incoming_type == msg_type]
        if type_results:
            matches = sum(1 for r in type_results if r.intent_match)
            exp_matches = sum(1 for r in type_results if r.expected_match)
            print(f"  {msg_type:12}: {len(type_results):2} samples | intent={matches}/{len(type_results)} ({matches/len(type_results)*100:3.0f}%) | exp_acc={exp_matches}/{len(type_results)} ({exp_matches/len(type_results)*100:3.0f}%)")

    # Show classification examples
    print(f"\n--- CLASSIFICATION EXAMPLES ---")
    for r in results[:10]:
        exp_marker = "✓" if r.expected_match else "✗"
        print(f"  \"{r.last_message[:35]:35}\" → {r.incoming_type:12} (expect {r.expected_responses}, gold={r.gold_intent}) {exp_marker}")

    # Sample outputs
    print(f"\n--- SAMPLE MATCHES ---")
    for r in [r for r in results if r.intent_match][:5]:
        print(f"  [{r.incoming_type}] \"{r.gold[:30]}\" → \"{r.generated[:30]}\"")

    print(f"\n--- SAMPLE MISMATCHES ---")
    for r in [r for r in results if not r.intent_match][:5]:
        print(f"  [{r.incoming_type}:{r.gold_intent}→{r.gen_intent}] \"{r.gold[:25]}\" → \"{r.generated[:25]}\"")

    # Save
    save_path = RESULTS_DIR / f"exp2_embedding_{model_name}_{n_samples}.json"
    with open(save_path, "w") as f:
        json.dump({
            "model": model_name,
            "n_samples": n,
            "elapsed": elapsed,
            "intent_match_rate": intent_matches / n,
            "expectation_accuracy": expected_matches / n,
            "results": [asdict(r) for r in results],
        }, f, indent=2)
    print(f"\nSaved: {save_path}")

    return intent_matches / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--model", type=str, default="lfm2.5-1.2b")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    run_experiment(args.model, args.samples, args.verbose)


if __name__ == "__main__":
    main()
