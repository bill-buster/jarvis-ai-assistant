# Research: Best Practices for Personalized Text Reply Generation

> **Compiled:** January 2026
> **Purpose:** Evidence-backed recommendations for improving JARVIS iMessage reply generation
> **Scope:** Academic research, industry practices, evaluation metrics, technical approaches, UX patterns

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Challenge Analysis](#challenge-analysis)
3. [Academic Research Findings](#academic-research-findings)
4. [Industry Practices](#industry-practices)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Technical Approaches](#technical-approaches)
7. [UX Patterns](#ux-patterns)
8. [Evidence-Backed Recommendations](#evidence-backed-recommendations)
9. [Implementation Priority](#implementation-priority)
10. [Sources](#sources)

---

## Executive Summary

This research addresses four core challenges in personalized reply generation:

| Challenge | Key Finding | Recommended Solution |
|-----------|-------------|---------------------|
| Model guessing when it should ask for more info | LLMs are trained to always answer; abstention is "an unsolved problem" | Implement confidence-based selective prediction with clarification prompts |
| Semantic similarity doesn't capture "many valid replies" | One-to-many problem requires multi-response generation | Use LLM-as-judge evaluation + diversity metrics; generate multiple options |
| Small models struggle with context | SLMs need efficient context management | Hybrid retrieval (RAG + recent context) with summarization |
| When to use few-shot vs fine-tuning vs RAG | Fine-tuning on new knowledge increases hallucinations | Few-shot for style, RAG for context, avoid fine-tuning for facts |

**Key Insight:** Research shows that fine-tuning LLMs on new knowledge linearly increases hallucination rates (Gekhman et al., 2024). The optimal approach combines few-shot learning for style imitation with RAG for contextual grounding.

---

## Challenge Analysis

### Challenge 1: Model Guessing vs. Asking for Clarification

**The Problem:** Models generate responses even when uncertain, leading to inappropriate or hallucinated replies.

**Research Evidence:**
> "LLMs are trained on datasets designed to elicit specific, correct answers, creating a powerful incentive to generate responses even when uncertain or lacking knowledge. The models are built to be maximally helpful, translating into a strong bias against abstention."
> — [Know Your Limits: A Survey of Abstention in Large Language Models](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00754/131566/Know-Your-Limits-A-Survey-of-Abstention-in-Large)

**Key Finding:** Even scaling model size brings little improvement in abstention behavior. Reasoning fine-tuning actually degrades abstention by 24% on average.

### Challenge 2: Semantic Similarity Metric Limitations

**The Problem:** Traditional metrics like BLEU and semantic similarity fail for open-ended generation because multiple valid responses exist.

**Research Evidence:**
> "Open-domain Dialogue (OD) exhibits a one-to-many (o2m) property, whereby multiple appropriate responses exist for a single dialogue context. This fundamental characteristic poses significant challenges for both generation and evaluation of dialogue systems."
> — [Modeling the One-to-Many Property in Open-Domain Dialogue with LLMs](https://arxiv.org/html/2506.15131v1)

> "BLEU does not consider the meanings of words, leading to penalties for synonyms or words with similar meanings that may be acceptable to humans."
> — [How to evaluate Text Generation Models](https://towardsdatascience.com/how-to-evaluate-text-generation-models-metrics-for-automatic-evaluation-of-nlp-models-e1c251b04ec1/)

### Challenge 3: Small Model Context Understanding

**The Problem:** On-device models (1-3B parameters) have limited context windows and reasoning capabilities.

**Research Evidence:**
> "Chain-of-thought prompting enables models to decompose multi-step problems into intermediate steps... models with fewer than 100 billion parameters often show no improvement when prompted with chains of thought."
> — [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)

> "As the context window grows, the model's performance starts to degrade... for many popular LLMs, performance degrades significantly as context length increases."
> — [LLM Context Management Guide](https://eval.16x.engineer/blog/llm-context-management-guide)

### Challenge 4: Few-shot vs. Fine-tuning vs. RAG

**The Problem:** Choosing the right personalization approach for style and context.

**Research Evidence:**
> "When large language models are aligned via supervised fine-tuning, they may encounter new factual information that was not acquired through pre-training... as the examples with new knowledge are eventually learned, they linearly increase the model's tendency to hallucinate."
> — [Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?](https://arxiv.org/abs/2405.05904)

---

## Academic Research Findings

### 1. Personalized Text Generation (Style Transfer)

**Key Papers:**
- [Evaluating Style-Personalized Text Generation](https://arxiv.org/html/2508.06374) - Identifies that LLM-based SPTG requires both n-gram and LLM-as-judges evaluation
- [TSTBench: A Comprehensive Benchmark](https://www.mdpi.com/1099-4300/27/6/575) - Establishes that style transfer must balance style strength, content preservation, and fluency

**Key Insights:**
1. **Style is implicit and personal:** "Current LLMs still struggle to reproduce nuanced personal styles—especially in informal and stylistically diverse domains" ([LLMs Struggle to Imitate Writing Styles](https://arxiv.org/html/2509.14543v1))

2. **Few-shot helps significantly:** "Across all models and datasets, the 5-shot setting consistently outperforms the 0-shot condition in authorship verification accuracy" (ibid.)

3. **Style sheet method:** Generate a persona description summarizing the author's style, then use it as a system prompt ([Personalizing Story Generation](https://arxiv.org/html/2502.13028))

**Application to JARVIS:**
- Extract user style features from message history
- Create a "style sheet" describing the user's tone, vocabulary, and patterns
- Use 5+ example messages in prompts for style consistency

### 2. PersonaChat and Conversational Personalization

**Key Datasets:**
- [PersonaChat](https://arxiv.org/pdf/1801.07243) - 4-5 sentence persona profiles for each interlocutor
- [Synthetic-Persona-Chat](https://arxiv.org/html/2312.10007v1) - 20k conversations with improved quality
- [HiCUPID](https://arxiv.org/html/2506.01262v1) - Benchmark for personalized assistants

**Key Insights:**
1. **Selective Prompting Tuning (SPT):** "Integrates context-prompt contrastive learning and prompt fusion learning to prevent repetitive responses" ([Selective Prompting Tuning](https://arxiv.org/html/2406.18187v1))

2. **Persona consistency matters:** Models must maintain persona across turns to feel authentic

**Application to JARVIS:**
- Build user persona profiles from message history
- Maintain persona consistency across reply suggestions
- Consider relationship-specific personas (formal with boss, casual with friends)

### 3. Abstention and Uncertainty

**Key Paper:** [Know Your Limits: A Survey of Abstention in Large Language Models](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00754/131566/Know-Your-Limits-A-Survey-of-Abstention-in-Large)

**Abstention Methods:**

| Method | Description | Effectiveness |
|--------|-------------|---------------|
| Confidence elicitation | Ask model "how confident are you?" | Limited - often miscalibrated |
| Sample consistency | Generate multiple responses, check agreement | Better - captures epistemic uncertainty |
| R-Tuning | Fine-tune on "I am unsure" responses | Teaches knowledge boundaries |
| Hybrid strategies | Combine multiple signals | Best results |

**Key Insight:** When confidence is low, the system should:
1. Identify uncertainty sources (insufficient context, ambiguity, capability gap)
2. Employ dynamic strategies (retrieval, clarification, or abstain)

**Application to JARVIS:**
- Generate multiple response candidates
- Measure consistency across candidates
- If low consistency: ask for clarification or offer options
- Never guess on ambiguous requests

### 4. Clarifying Questions in Dialogue

**Key Research:**
- [ClariQ Dataset](https://github.com/aliannejadi/ClariQ) - When to ask clarifying questions
- [Asking Clarification Questions for Task-Oriented Dialogues](https://arxiv.org/abs/2305.13690)

**Key Insights:**
1. **Two sources of ambiguity:**
   - User's inability to describe complex information needs
   - Missing/ambiguous information about user preferences

2. **Smart clarification:** "When asked for help generating a recipe, the bot requested further details to clarify the user's needs before proceeding" ([NN/G Conversation Types](https://www.nngroup.com/articles/AI-conversation-types/))

**Application to JARVIS:**
- Detect ambiguous conversation contexts
- Suggest clarifying questions with potential answers
- Example: "What should I say about [topic]? Do you want to accept, decline, or ask for more info?"

---

## Industry Practices

### 1. Google Smart Reply

**How It Works:**
> "Smart Reply utilizes one neural network to break down and encode the received email, while a separate network predicts responses. The goal is to identify 'thought vectors' that capture the gist without getting hung up on diction."
> — [Efficient Smart Reply for Gmail](https://research.google/blog/efficient-smart-reply-now-for-gmail/)

**Key Features:**
- **Hierarchy of modules:** Process at different temporal scales
- **Personalization:** "If you're more of a 'thanks!' than a 'thanks.' person, it will suggest the response that's more like you"
- **Three options:** Always presents 3 diverse replies
- **Gemini upgrade:** Now provides contextual Smart Replies considering full thread

**Lessons for JARVIS:**
- Always generate multiple options (3 is optimal)
- Learn user preferences from acceptance patterns
- Consider full conversation context, not just last message

### 2. LinkedIn AI Writing Assistant

**Key Features:**
- Tone selection (Casual, Professional, Empathetic, Persuasive)
- Goal-based responses (introduce yourself, ask about experience, etc.)
- Context-aware suggestions
- Always requires user review and editing

**Lessons for JARVIS:**
- Let users select response intent/tone
- Make it clear the response is a draft to edit
- Support relationship-specific presets

### 3. Commercial Chatbot Best Practices

**Key Practices from Enterprise Deployments:**

| Practice | Benefit | Source |
|----------|---------|--------|
| Integrate with user data | Personalized responses | [Kommunicate](https://www.kommunicate.io/blog/chatbot-personalization/) |
| Sentiment analysis | Detect frustration, adjust tone | [Bird](https://bird.com/en-us/blog/how-to-build-a-personalized-ai-chatbot-experience-for-your-customers) |
| Seamless human handoff | Handle complexity gracefully | [Salesforce](https://www.salesforce.com/agentforce/chatbot/best-practices/) |
| Balance personalization with privacy | Build trust | [ChatBot.com](https://www.chatbot.com/academy/chatbot-designer-free-course/personalization/) |

**Application to JARVIS:**
- Use message history for personalization (already local)
- Detect emotional context of incoming messages
- Know when NOT to generate (sensitive topics, complex situations)

### 4. AI Writing Assistants (Grammarly, Jasper)

**Grammarly Approach:**
> "Grammarly's AI has the unique ability to personalize suggestions based on what you're writing and who's reading it"
> — [Grammarly AI Writing Assistant](https://www.grammarly.com/ai-writing-assistant)

**Jasper Approach:**
> "As you use it more frequently, it naturally picks up on your tone and adapts to make the material sound more personalized"
> — [Jasper AI Review](https://www.onsaas.me/blog/jasper-ai-review)

**Key Insight:** Both focus on audience awareness and adaptive learning, not just style mimicry.

---

## Evaluation Metrics

### The Problem with Traditional Metrics

| Metric | Limitation | Better Alternative |
|--------|------------|-------------------|
| BLEU | Penalizes synonyms, requires exact match | BERTScore (0.93 vs 0.70 correlation with humans) |
| Semantic similarity | Single reference, ignores diversity | Multi-reference or reference-free |
| Perplexity | Measures fluency, not appropriateness | LLM-as-judge |

### Recommended Evaluation Framework

#### 1. **BERTScore for Semantic Quality**
> "BERTScore achieved a 0.93 Pearson correlation with human judgments, significantly outperforming BLEU (0.70) and ROUGE (0.78)"
> — [BERTScore Explained](https://galileo.ai/blog/bert-score-explained-guide)

**How it works:** Uses contextual embeddings to compare semantic similarity, tolerates synonyms and paraphrasing.

#### 2. **LLM-as-Judge for Multi-Criteria**
> "G-Eval uses LLMs to evaluate LLM outputs and is one of the best ways to create task-specific metrics"
> — [LLM Evaluation Metrics Guide](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)

**Criteria to evaluate:**
- Style match (does it sound like the user?)
- Appropriateness (is it suitable for this relationship/context?)
- Helpfulness (does it move the conversation forward?)
- Naturalness (does it feel human?)

#### 3. **Diversity Metrics**
> "Metrics like Distinct-n and Self-BLEU track repetitiveness or variety in generated text"
> — [RAG Evaluation Metrics](https://www.elastic.co/search-labs/blog/evaluating-rag-metrics)

**Why it matters:** Multiple valid responses should be diverse, not minor variations.

#### 4. **Reference-Free Factuality**
> "SelfCheckGPT assumes hallucinated outputs are not reproducible, whereas if an LLM has knowledge of a given concept, sampled responses are likely to be similar and contain consistent facts"
> — [LLM Evaluation Metrics](https://arya.ai/blog/llm-evaluation-metrics)

#### 5. **Human Evaluation Best Practices**
> "Use continuous rating scales (visual analog scale) rather than Likert scales—research shows they yield more consistent results for dialog system evaluation"
> — [Survey on Evaluation Methods for Dialogue Systems](https://pmc.ncbi.nlm.nih.gov/articles/PMC7817575/)

**Practical approach:**
1. A/B test generated replies vs. actual user replies
2. Measure if recipients can detect AI-generated responses
3. Track which suggestions users accept vs. edit vs. reject

### Handling the One-to-Many Problem

**Research Solution:** PLATO model uses discrete latent variables to capture different valid response types.

**Practical Solution:** Generate N diverse responses, let user choose. Evaluate whether user's actual reply is semantically close to ANY generated option.

---

## Technical Approaches

### 1. Few-shot Learning vs. Fine-tuning

**Clear Winner: Few-shot for Style, Avoid Fine-tuning for Facts**

| Approach | Use Case | Caveats |
|----------|----------|---------|
| Few-shot | Style imitation, tone matching | Needs good examples, limited by context window |
| Fine-tuning | When you have 10k+ examples, need production latency | Increases hallucinations for new knowledge |
| RAG | Contextual grounding, conversation history | Requires good retrieval, adds latency |

**Evidence:**
> "Fine-tuning works well when the fine-tuning dataset consists primarily of examples that are known to the pre-trained LLM. Conversely, fine-tuning on a dataset with a higher proportion of examples containing new knowledge results in decreased performance and a higher tendency to hallucinate."
> — [Fine-tuning Hallucinations in LLMs](https://rewirenow.com/en/resources/blog/fine-tuning-hallucinations-in-llms/)

**Recommendation for JARVIS:**
```
Style matching: 5-shot examples of user's messages
Context: RAG from recent conversation history
Facts: Never fine-tune, use retrieval instead
```

### 2. RAG Best Practices for Personalization

**Key Practices:**
1. **Hybrid search:** Combine lexical and vector retrieval
2. **Re-ranking:** Use cross-encoder to select most relevant context
3. **Query augmentation:** Expand ambiguous queries before retrieval
4. **Chunk wisely:** Messages are natural units, no need for complex chunking

> "By integrating retrieval mechanisms, RAG systems fetch relevant external knowledge during the generation process, ensuring the model's output is informed by up-to-date and contextually relevant information"
> — [Searching for Best Practices in RAG](https://aclanthology.org/2024.emnlp-main.981/)

**For JARVIS:**
- Retrieve recent messages in conversation (last 10-20)
- Retrieve similar past conversations with same contact
- Retrieve user's typical responses to similar message types

### 3. Small Language Model Optimization

**Recommended Models for On-Device:**
- **Qwen2-0.5B/1.5B:** Optimized for portable devices ([HuggingFace SLM Overview](https://huggingface.co/blog/jjokah/small-language-model))
- **Phi-3-mini (3.8B):** Matches GPT-3.5 on benchmarks, runs on mobile
- **TinyLLama (1.1B):** Strong for conversational tasks

**Optimization Techniques:**
1. **Quantization:** 4-bit quantization reduces memory 4x with minimal quality loss
2. **Knowledge distillation:** Train small model from large model outputs
3. **Efficient context management:** Summarize older context, keep recent verbatim

### 4. Chain-of-Thought for Better Generation

**Limitation for Small Models:**
> "Models with fewer than 100 billion parameters often show no improvement when prompted with chains of thought"
> — [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)

**Alternative for Small Models:**
- **Structured prompts:** Guide the model step-by-step without expecting it to reason
- **Template-based generation:** Fill in slots rather than free generation
- **Multi-turn refinement:** Generate draft, then refine

### 5. Context Window Management

**Best Practices:**
> "Only include the most relevant recent exchanges along with a summary of older parts of the conversation"
> — [GenAI Context History Best Practices](https://verticalserve.medium.com/genai-managing-context-history-best-practices-a350e57cc25f)

| Strategy | When to Use |
|----------|-------------|
| Keep last N messages verbatim | Always (N=5-10) |
| Summarize older context | Conversations > 20 messages |
| Cache conversation summaries | Frequently accessed contacts |
| Start fresh for new topics | Topic detection shows shift |

---

## UX Patterns

### 1. When to Ask vs. Generate

**Pattern: Ask When Uncertain, Suggest Answers**

> "When you have AI ask clarifying questions, have it suggest the answers too. Example: 'Ask me clarifying questions and suggest answers based on the context.'"
> — [7 Prompt UX Patterns](https://blog.mariohayashi.com/p/7-prompt-ux-patterns-to-help-you)

**Implementation:**
```
User: What should I say to Mom about dinner?

BAD: "I need more information. What do you want to say?"

GOOD: "I can suggest a reply. What's your intent?
  - Accept the invitation
  - Decline politely
  - Ask about timing
  - Something else..."
```

### 2. Multiple Options Pattern

> "Instead of giving one answer, provide alternative responses so users can choose the best fit"
> — [Designing Trustworthy AI Assistants](https://www.mtlc.co/designing-trustworthy-ai-assistants-9-simple-ux-patterns-that-make-a-big-difference/)

**Design Guidelines:**
- Show 3 options (Google Smart Reply's proven pattern)
- Make options genuinely diverse (different tones, lengths, or approaches)
- Allow quick selection with edit capability
- Learn from which options users select

### 3. Confidence Signaling

> "Show confidence scores visually (bars, badges), and use clear language to flag low-confidence outputs. Clarify uncertainty with suggested alternatives."
> — [How to Design AI UIs for Confidence](https://wild.codes/candidate-toolkit-question/how-to-design-ai-uis-that-show-confidence-uncertainty-trust)

**Implementation:**
- Don't show raw confidence scores (confusing)
- Use language: "Here's a suggestion" vs "I'm not sure, but..."
- Provide edit prompts: "You might want to adjust the tone"

### 4. Graceful Failure and Recovery

> "When users encounter an error or a bad suggestion, they shouldn't hit a dead end. Instead, the interface should offer a path forward."
> — [Embrace AI's Uncertainty in UX](https://www.uxtigers.com/post/ai-uncertainty-ux)

**Fallback Strategies:**
1. If low confidence: "I'm not sure. Would you like to give me more context?"
2. If topic is sensitive: "This seems personal. Want to write this yourself?"
3. If all else fails: "Here are some starting points you can edit..."

### 5. Authorship Preservation

> "When suggestions echoed the writer's register and cadence, they were easier to adopt and felt more like extensions of the writer's intent. When suggestions defaulted to generic professional rhetoric, they contributed to the sense that the draft reflected the assistant's voice rather than the writer's."
> — [Design Patterns for Preserving Authorship](https://arxiv.org/html/2601.10236v1)

**Key Patterns:**
1. **On-demand initiation:** User triggers suggestions, not automatic
2. **Micro-suggestions:** Complete words/phrases, not full messages
3. **Voice anchoring:** Match user's register and vocabulary
4. **Point-of-decision provenance:** Show what's AI vs. user

---

## Evidence-Backed Recommendations

### Recommendation 1: Implement Selective Generation with Clarification

**Problem Addressed:** Model guessing when uncertain

**Implementation:**
```python
def should_generate(context, confidence_threshold=0.7):
    # Generate multiple candidate responses
    candidates = generate_n_responses(context, n=5)

    # Measure consistency across candidates
    consistency_score = measure_semantic_consistency(candidates)

    if consistency_score < confidence_threshold:
        # Ask for clarification instead of guessing
        return "clarify", generate_clarification_options(context)
    else:
        # Return diverse subset of consistent responses
        return "generate", select_diverse_top_k(candidates, k=3)
```

**Evidence:** Sample consistency is more reliable than confidence elicitation for detecting uncertainty.

### Recommendation 2: Multi-Response Generation with User Selection

**Problem Addressed:** One-to-many problem, semantic similarity limitations

**Implementation:**
- Always generate 3+ diverse responses
- Use different "intents" or "tones" as diversity seeds
- Track which options users select for implicit feedback
- Evaluate against whether user's actual response is close to ANY option

**Evidence:** Google Smart Reply's success with 3-option pattern; PLATO's discrete latent variables.

### Recommendation 3: Hybrid Retrieval for Context

**Problem Addressed:** Small models struggle with context

**Implementation:**
```python
def build_context(message, contact):
    # Recent context (verbatim)
    recent = get_last_n_messages(contact, n=10)

    # Similar past exchanges (RAG)
    similar = vector_search(message, contact_history, k=3)

    # User style examples (few-shot)
    style_examples = get_user_response_examples(contact, n=5)

    # Summarized relationship context
    relationship_summary = get_relationship_summary(contact)

    return build_prompt(recent, similar, style_examples, relationship_summary)
```

**Evidence:** Hybrid search outperforms single-method retrieval; summarization prevents context overload.

### Recommendation 4: Few-shot for Style, RAG for Facts

**Problem Addressed:** Fine-tuning increases hallucinations

**Implementation:**
- Create user "style sheet" from message analysis
- Include 5 example user responses in every prompt
- Use RAG for conversation context and facts
- Never fine-tune on user-specific data

**Evidence:** Gekhman et al. (2024) shows linear increase in hallucinations from fine-tuning on new knowledge.

### Recommendation 5: LLM-as-Judge Evaluation

**Problem Addressed:** Semantic similarity doesn't work for many-valid-answers

**Implementation:**
```python
def evaluate_response(generated, context, user_style):
    criteria = {
        "style_match": "Does this sound like the user?",
        "appropriateness": "Is this suitable for this relationship?",
        "helpfulness": "Does this move the conversation forward?",
        "naturalness": "Does this feel human?"
    }

    # Use LLM to evaluate each criterion
    scores = {}
    for name, question in criteria.items():
        scores[name] = llm_evaluate(generated, context, question)

    return scores
```

**Evidence:** G-Eval and LLM-as-judge approaches correlate better with human judgment than n-gram metrics.

### Recommendation 6: Relationship-Aware Personalization

**Problem Addressed:** One style doesn't fit all relationships

**Implementation:**
- Detect relationship type (family, friend, colleague, etc.)
- Maintain per-contact style profiles
- Adjust formality, emoji usage, message length by relationship
- Learn from user corrections

**Evidence:** LinkedIn's tone selection; enterprise chatbot personalization practices.

---

## Implementation Priority

### Phase 1: Foundation (Highest Impact, Lowest Risk)

| Feature | Effort | Impact | Evidence |
|---------|--------|--------|----------|
| Multi-response generation (3 options) | Medium | High | Google Smart Reply success |
| Few-shot style examples in prompts | Low | High | 5-shot consistently outperforms 0-shot |
| Recent conversation context | Low | High | Context is essential for coherence |

### Phase 2: Uncertainty Handling (Medium Impact)

| Feature | Effort | Impact | Evidence |
|---------|--------|--------|----------|
| Consistency-based confidence | Medium | Medium | More reliable than raw confidence |
| Clarification with suggested answers | Medium | High | Better UX than generic "I don't know" |
| Intent detection (accept/decline/ask) | Medium | Medium | Reduces ambiguity |

### Phase 3: Advanced Personalization (Higher Effort)

| Feature | Effort | Impact | Evidence |
|---------|--------|--------|----------|
| User style sheet generation | High | High | Personalizing story generation research |
| Per-contact profiles | High | Medium | Relationship-aware responses |
| RAG from similar past conversations | High | Medium | Contextual grounding |

### Phase 4: Evaluation and Learning (Ongoing)

| Feature | Effort | Impact | Evidence |
|---------|--------|--------|----------|
| LLM-as-judge evaluation pipeline | Medium | Medium | Better than BLEU/similarity |
| Implicit feedback from selections | Low | High | Learn which options work |
| A/B testing framework | High | High | Measure real-world impact |

---

## Sources

### Academic Papers
- [Know Your Limits: A Survey of Abstention in Large Language Models](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00754/131566/Know-Your-Limits-A-Survey-of-Abstention-in-Large) - TACL 2025
- [Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?](https://arxiv.org/abs/2405.05904) - EMNLP 2024
- [Modeling the One-to-Many Property in Open-Domain Dialogue with LLMs](https://arxiv.org/html/2506.15131v1) - 2025
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) - NeurIPS 2022
- [Catch Me If You Can? LLMs Still Struggle to Imitate Writing Styles](https://arxiv.org/html/2509.14543v1) - EMNLP 2025
- [Searching for Best Practices in Retrieval-Augmented Generation](https://aclanthology.org/2024.emnlp-main.981/) - EMNLP 2024
- [Who Owns the Text? Design Patterns for Preserving Authorship](https://arxiv.org/html/2601.10236v1) - 2026
- [PLATO: Pre-trained Dialogue Generation Model with Discrete Latent Variable](https://arxiv.org/abs/1910.07931) - ACL 2020
- [Evaluating Style-Personalized Text Generation](https://arxiv.org/html/2508.06374) - 2025
- [Survey on Evaluation Methods for Dialogue Systems](https://pmc.ncbi.nlm.nih.gov/articles/PMC7817575/) - AI Review 2021

### Industry Resources
- [Efficient Smart Reply, Now for Gmail](https://research.google/blog/efficient-smart-reply-now-for-gmail/) - Google Research
- [Smart Reply: Automated Response Suggestion for Email](https://research.google/pubs/pub45189/) - Google Research
- [Grammarly AI Writing Assistant](https://www.grammarly.com/ai-writing-assistant) - Grammarly
- [LinkedIn AI-powered Writing Assistant](https://www.linkedin.com/help/linkedin/answer/a1444194) - LinkedIn Help
- [RPLY: AI Assistant for iMessage](https://techcrunch.com/2025/02/06/rply-is-a-new-ai-assistant-that-responds-to-missed-texts/) - TechCrunch

### UX and Design
- [7 Prompt UX Patterns](https://blog.mariohayashi.com/p/7-prompt-ux-patterns-to-help-you) - UX Guide
- [Embrace AI's Uncertainty in UX](https://www.uxtigers.com/post/ai-uncertainty-ux) - UX Tigers
- [Design Patterns For AI Interfaces](https://www.smashingmagazine.com/2025/07/design-patterns-ai-interfaces/) - Smashing Magazine
- [The 6 Types of Conversations with Generative AI](https://www.nngroup.com/articles/AI-conversation-types/) - Nielsen Norman Group
- [Designing Trustworthy AI Assistants](https://www.mtlc.co/designing-trustworthy-ai-assistants-9-simple-ux-patterns-that-make-a-big-difference/) - Mass Tech Leadership Council

### Evaluation and Metrics
- [BERTScore in AI: Enhancing Text Evaluation](https://galileo.ai/blog/bert-score-explained-guide) - Galileo AI
- [LLM Evaluation Metrics: The Ultimate Guide](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation) - Confident AI
- [RAG Evaluation Metrics](https://www.elastic.co/search-labs/blog/evaluating-rag-metrics) - Elasticsearch Labs

### Technical Guides
- [LLM Context Management Guide](https://eval.16x.engineer/blog/llm-context-management-guide) - 16x Engineer
- [GenAI Context History Best Practices](https://verticalserve.medium.com/genai-managing-context-history-best-practices-a350e57cc25f) - Medium
- [Small Language Models Overview](https://huggingface.co/blog/jjokah/small-language-model) - Hugging Face
- [Fine-tuning vs Few-shot Learning](https://labelbox.com/guides/zero-shot-learning-few-shot-learning-fine-tuning/) - Labelbox

---

## Appendix: Key Quotes for Reference

### On Abstention
> "Abstention is the refusal to answer a query. When a model fully abstains, it may begin a response with 'I don't know' or refuse to answer in another way. In reality, abstention encompasses a spectrum of behaviors, e.g., expressing uncertainty, providing conflicting conclusions, or refusing due to potential harm are all forms of abstention."

### On Fine-tuning
> "These results highlight the risk in introducing new factual knowledge through fine-tuning, and support the view that large language models mostly acquire factual knowledge through pre-training, whereas fine-tuning teaches them to use it more efficiently."

### On Style Imitation
> "Across all models and datasets, the 5-shot setting consistently outperforms the 0-shot condition in authorship verification (AV) accuracy. This shows that providing even a few writing examples significantly improves an LLM's ability to imitate implicit personal writing style."

### On Evaluation
> "BLEU, ROUGE, and METEOR metrics measure word overlap between generated and reference responses. Many researchers argue these metrics are not appropriate for evaluation of open-domain dialog agents since there are many plausible responses to the same user's input."

### On UX
> "When suggestions echoed the writer's register and cadence, they were easier to adopt (higher incorporation) and felt more like extensions of the writer's intent (higher ownership)."
