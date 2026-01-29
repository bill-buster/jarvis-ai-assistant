# Testing Guide for JARVIS v3

## Philosophy: Test Behavior, Not Implementation

**Good Test**: "When I send 'wanna hang?', I get 3 reply suggestions"
**Bad Test**: "When I call _parse_replies with strip_emojis=True, it returns..."

## Test Levels

### Level 1: Unit Tests (89% of tests)
**Purpose**: Verify individual components work in isolation
**Speed**: <1 second
**No model loading**: Everything is mocked

```python
# Example: Testing ReplyGenerator without loading MLX
def test_can_create_reply_generator_mocked(mock_model_loader):
    """Test that ReplyGenerator can be instantiated."""
    from core.generation.reply_generator import ReplyGenerator
    
    # Create generator with mocked loader
    generator = ReplyGenerator(mock_model_loader)
    
    # Verify it has the right methods
    assert hasattr(generator, 'generate_replies')
    assert hasattr(generator, '_find_past_replies')
    assert hasattr(generator, '_find_cross_conversation_replies')
```

**When to write**: 
- New public method
- Complex logic that needs validation
- Edge cases (empty input, invalid data)

**Run them**:
```bash
make test-v3    # 17 tests, 0.36s
```

---

### Level 2: Integration Tests (10% of tests)
**Purpose**: Verify components work together
**Speed**: <10 seconds
**Partial mocking**: Some real services, some mocked

```python
# Example: Testing API endpoint with mocked iMessage
def test_generate_replies_endpoint(client, mock_message_reader):
    """Test the /generate/replies endpoint."""
    response = client.post("/generate/replies", json={
        "chat_id": "test-chat-123"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "replies" in data
    assert len(data["replies"]) == 3
```

**When to write**:
- API endpoints
- Service integration
- Database interactions

**Run them**:
```bash
make test    # All tests including integration
```

---

### Level 3: E2E / Manual Tests (1% of tests)
**Purpose**: Validate entire flow with real model
**Speed**: 30-60 seconds (model loading time)
**No mocking**: Real MLX model, real iMessage DB

```python
# Example: Manual test with real model
def test_real_generation():
    """Test with actual model (slow, run manually)."""
    from core.models.loader import ModelLoader
    from core.generation.reply_generator import ReplyGenerator
    
    # This loads the real 0.5GB model
    loader = ModelLoader('lfm2.5-1.2b')
    generator = ReplyGenerator(loader)
    
    result = generator.generate_replies(
        messages=['Hey want to grab dinner tonight?'],
        chat_id='test-chat'
    )
    
    assert len(result.replies) == 3
    for reply in result.replies:
        assert len(reply.text) > 0
        print(f"Generated: {reply.text}")
```

**When to run**:
- Before committing major changes
- Validating RAG improvements
- Checking model quality

**Run them**:
```bash
# Manual test with real model
uv run python -c "
from core.generation.reply_generator import ReplyGenerator
from core.models.loader import ModelLoader

print('Loading model...')
loader = ModelLoader()
gen = ReplyGenerator(loader)

print('Generating replies...')
result = gen.generate_replies(['Want to hang?'], 'test')

print('\nResults:')
for i, r in enumerate(result.replies, 1):
    print(f'{i}. {r.text}')
"
```

---

## Current Test Suite

```
v3/tests/
├── test_basic.py (7 tests)
│   ├── test_imports_work
│   ├── test_model_registry_has_correct_model
│   ├── test_can_create_reply_generator_mocked
│   ├── test_context_analyzer_detects_intent
│   ├── test_style_analyzer_analyzes_messages
│   ├── test_prompt_building
│   └── test_relationship_registry_basic
│
└── test_relationship_registry.py (10 tests)
    ├── test_load_profiles
    ├── test_get_relationship_by_name
    ├── test_get_relationship_by_phone
    ├── test_get_relationship_not_found
    ├── test_get_relationship_from_chat_id_imessage
    ├── test_get_relationship_from_chat_id_unknown
    ├── test_get_contacts_by_category_friend
    ├── test_get_contacts_by_category_empty
    ├── test_missing_profiles_file
    └── test_phone_normalization
```

**All 17 tests pass in 0.36 seconds** ✅

---

## Testing the RAG System

### Test 1: Same-Conversation Search

```python
def test_same_conversation_search():
    """Verify RAG finds replies from the same chat."""
    from core.embeddings import get_embedding_store
    
    store = get_embedding_store()
    
    # Add some test messages
    store.add_messages([
        {
            'message_id': '1',
            'chat_id': 'test-chat',
            'text': 'Want to grab dinner?',
            'sender': 'friend',
            'is_from_me': False,
        },
        {
            'message_id': '2', 
            'chat_id': 'test-chat',
            'text': 'sure!',
            'sender': 'me',
            'is_from_me': True,
        }
    ])
    
    # Search for similar to "Want to grab dinner?"
    results = store.find_your_past_replies(
        incoming_message='Want to get food?',
        chat_id='test-chat',
        limit=5
    )
    
    # Should find "sure!" as a reply
    assert len(results) > 0
    assert any('sure' in r[1].lower() for r in results)
```

### Test 2: Cross-Conversation Search

```python
def test_cross_conversation_search():
    """Verify RAG finds replies from similar relationships."""
    from core.embeddings import get_embedding_store
    from core.embeddings.relationship_registry import get_relationship_registry
    
    store = get_embedding_store()
    registry = get_relationship_registry()
    
    # Profile two friends
    registry.register_profile('Friend A', 'close_friend', ['+123'])
    registry.register_profile('Friend B', 'close_friend', ['+456'])
    
    # Add messages for Friend A
    store.add_messages([
        {
            'message_id': '1',
            'chat_id': 'chat-a',
            'text': 'Want to hang?',
            'sender': 'Friend A',
            'is_from_me': False,
        },
        {
            'message_id': '2',
            'chat_id': 'chat-a', 
            'text': 'yeah down',
            'sender': 'me',
            'is_from_me': True,
        }
    ])
    
    # Search from Friend B's perspective
    results = store.search_cross_conversation(
        query='Want to hang out?',
        relationship_type='close_friend',
        exclude_chat_id='chat-b',
        limit=5
    )
    
    # Should find "yeah down" from Friend A's chat
    assert len(results) > 0
    assert any('down' in r[1].lower() for r in results)
```

---

## Evaluating Reply Quality

### Manual Evaluation (Gold Standard)

```python
# Generate test samples
test_cases = [
    {
        'incoming': 'Want to grab dinner tonight?',
        'context': ['Friend: Want to grab dinner tonight?'],
        'gold_response': 'sure!',
        'relationship': 'close_friend'
    },
    # ... 29 more cases
]

# Generate replies
generated = []
for case in test_cases:
    result = generator.generate_replies(
        case['context'], 
        relationship=case['relationship']
    )
    generated.append({
        'incoming': case['incoming'],
        'gold': case['gold_response'],
        'generated': result.replies[0].text
    })

# Save for human evaluation
import json
with open('evaluation_samples.json', 'w') as f:
    json.dump(generated, f, indent=2)
```

### Automated Metrics

```python
def evaluate_intent_match(generated: str, gold: str) -> bool:
    """Check if generated and gold have same intent."""
    from scripts.validate_classifier import IntentClassifier
    
    clf = IntentClassifier()
    gen_intent = clf.classify(generated)
    gold_intent = clf.classify(gold)
    
    return gen_intent == gold_intent


def evaluate_embedding_similarity(generated: str, gold: str) -> float:
    """Check semantic similarity using embeddings."""
    from core.embeddings.model import get_embedding_model
    
    model = get_embedding_model()
    gen_emb = model.embed(generated)
    gold_emb = model.embed(gold)
    
    # Cosine similarity
    similarity = np.dot(gen_emb, gold_emb) / (np.linalg.norm(gen_emb) * np.linalg.norm(gold_emb))
    return float(similarity)


# Run evaluation
results = []
for g in generated:
    intent_match = evaluate_intent_match(g['generated'], g['gold'])
    similarity = evaluate_embedding_similarity(g['generated'], g['gold'])
    
    results.append({
        'incoming': g['incoming'],
        'intent_match': intent_match,
        'similarity': similarity,
        'generated': g['generated'],
        'gold': g['gold']
    })

# Calculate metrics
total = len(results)
intent_accuracy = sum(1 for r in results if r['intent_match']) / total
avg_similarity = sum(r['similarity'] for r in results) / total

print(f"Intent Match: {intent_accuracy:.1%}")
print(f"Avg Similarity: {avg_similarity:.3f}")
```

---

## Debugging Failed Tests

### Test Isolation

```python
# Bad: Tests depend on each other
def test_a():
    global_state['value'] = 5  # Mutates global state

def test_b():
    assert global_state['value'] == 5  # Depends on test_a!

# Good: Each test is independent
def test_a():
    state = {'value': 5}  # Local state
    assert state['value'] == 5

def test_b():
    state = {'value': 10}  # Different local state
    assert state['value'] == 10
```

### Using Fixtures

```python
import pytest

@pytest.fixture
def fresh_store(tmp_path):
    """Create a fresh embedding store for each test."""
    from core.embeddings.store import EmbeddingStore
    return EmbeddingStore(db_path=tmp_path / "test.db")

@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        {
            'message_id': '1',
            'chat_id': 'test',
            'text': 'Hello',
            'sender': 'friend',
            'is_from_me': False,
        }
    ]

def test_add_messages(fresh_store, sample_messages):
    """Test adding messages with fresh store."""
    fresh_store.add_messages(sample_messages)
    assert fresh_store.count() == 1
```

### Debugging Tips

```bash
# Run single test with verbose output
uv run pytest tests/test_basic.py::test_imports_work -vvs

# Run with debugging
uv run pytest tests/ --pdb  # Drop into debugger on failure

# Run with coverage
uv run pytest tests/ --cov=core --cov-report=html

# Run specific test file
uv run pytest tests/test_relationship_registry.py -v
```

---

## Testing Checklist

Before committing code:

- [ ] `make test-v3` passes (17 tests, <1s)
- [ ] `make lint` shows no errors
- [ ] New code has tests
- [ ] Tests mock expensive operations (model loading, DB access)
- [ ] Tests are independent (no shared state)
- [ ] Tests describe behavior, not implementation

Before releasing:

- [ ] `make test` passes (all tests)
- [ ] Manual test with real model works
- [ ] Human evaluation shows improvement
- [ ] Performance is acceptable (<3s generation time)

---

## Common Testing Patterns

### Pattern 1: Testing the Golden Path

```python
def test_generate_replies_success():
    """Test successful reply generation."""
    # Setup
    generator = ReplyGenerator(mock_loader)
    messages = [{'text': 'Want to hang?', 'sender': 'friend', 'is_from_me': False}]
    
    # Execute
    result = generator.generate_replies(messages, 'test-chat')
    
    # Verify
    assert len(result.replies) == 3
    assert all(r.text for r in result.replies)
    assert result.generation_time_ms > 0
```

### Pattern 2: Testing Edge Cases

```python
def test_generate_replies_empty_messages():
    """Test with empty message list."""
    generator = ReplyGenerator(mock_loader)
    
    with pytest.raises(ValueError, match="No messages provided"):
        generator.generate_replies([], 'test-chat')


def test_generate_replies_no_incoming():
    """Test when all messages are from me."""
    generator = ReplyGenerator(mock_loader)
    messages = [{'text': 'Hello', 'sender': 'me', 'is_from_me': True}]
    
    result = generator.generate_replies(messages, 'test-chat')
    # Should still work, just no context
    assert len(result.replies) == 3
```

### Pattern 3: Testing with Mocks

```python
def test_rag_uses_embeddings(mock_embedding_store):
    """Test that RAG calls embedding store."""
    generator = ReplyGenerator(mock_loader)
    generator._embedding_store = mock_embedding_store
    
    messages = [{'text': 'Test', 'sender': 'friend', 'is_from_me': False}]
    generator.generate_replies(messages, 'test-chat')
    
    # Verify embedding store was called
    mock_embedding_store.find_your_past_replies.assert_called_once()
```

---

## Next Steps

1. **Run existing tests**: `make test-v3`
2. **Add tests for new features** as you build them
3. **Evaluate quality** using the manual + automated approach
4. **Iterate** based on evaluation results

Remember: **Tests are documentation**. Write them so future you understands what the code should do.
