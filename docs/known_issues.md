# JARVIS Known Issues and Limitations

**Date**: 2026-01-30
**Version**: Post-Consolidation

---

## Platform Requirements

### macOS Only

JARVIS requires macOS because:
- iMessage database (`~/Library/Messages/chat.db`) is macOS-specific
- MLX acceleration requires Apple Silicon
- AddressBook contacts integration is macOS-specific
- Calendar integration uses macOS Calendar database

**Will NOT work on**: Linux, Windows, Intel Macs

### Memory Requirements

| Model | Min RAM | Recommended |
|-------|---------|-------------|
| Qwen2.5-0.5B-4bit | 4GB | 6GB |
| Qwen2.5-1.5B-4bit | 6GB | 8GB (default) |
| Qwen2.5-3B-4bit | 8GB | 12GB |

The default model (Qwen2.5-1.5B-Instruct-4bit) targets 8GB MacBook Air.

---

## Known Issues

### HIGH Priority

#### 1. iMessage Database Access Requires Full Disk Access

**Symptom**: "Permission denied" or empty conversation list

**Cause**: macOS Sequoia+ requires explicit Full Disk Access for applications reading `~/Library/Messages/chat.db`

**Solution**:
1. Open System Settings > Privacy & Security > Full Disk Access
2. Add your Terminal app (or IDE)
3. Restart the Terminal/IDE

**Verification**: `python -m jarvis.setup --check`

#### 2. Model Download Required Before First Use

**Symptom**: "FileNotFoundError: Model not found"

**Cause**: MLX models must be downloaded from HuggingFace before use

**Solution**:
```bash
huggingface-cli download mlx-community/Qwen2.5-1.5B-Instruct-4bit
```

Or run setup wizard:
```bash
python -m jarvis.setup
```

#### 3. First Generation is Slow (Cold Start)

**Symptom**: First reply takes 10-15 seconds

**Cause**: Model loading and Metal shader compilation on first run

**Solution**: This is expected behavior. Subsequent generations use cached model (~2-3s).

**Mitigation**: The API server pre-loads the model on startup.

### MEDIUM Priority

#### 4. iMessage Sender is Unreliable (DEPRECATED)

**Symptom**: Sending messages fails or requires constant permission prompts

**Cause**: Apple restricts AppleScript automation for Messages.app. Known issues:
- Requires Automation permission
- May be blocked by SIP
- Requires Messages.app to be running
- Breaks with macOS updates

**Solution**: The `IMessageSender` class is deprecated. JARVIS generates reply suggestions but does NOT send them automatically.

**Location**: `integrations/imessage/sender.py` (marked DEPRECATED)

#### 5. Group Chat Handling is Limited

**Symptom**: Reply suggestions for group chats may be less accurate

**Cause**:
- Template matching uses group size but context is still limited
- Multiple participant threads are harder to track
- RAG search doesn't distinguish group context well

**Mitigation**: The intent classifier has GROUP_COORDINATION, GROUP_RSVP, GROUP_CELEBRATION intents, but quality varies.

#### 6. HHEM Model Requires Separate Download

**Symptom**: HHEM benchmark fails with model not found

**Cause**: Vectara HHEM model must be downloaded separately

**Solution**:
```bash
huggingface-cli download vectara/hallucination_evaluation_model
```

### LOW Priority

#### 7. Contact Resolution May Miss Some Contacts

**Symptom**: Phone numbers shown instead of names

**Cause**: AddressBook database structure varies across macOS versions and sync sources (iCloud, Google, etc.)

**Workaround**: JARVIS tries multiple AddressBook sources but may miss contacts synced from certain providers.

#### 8. Schema Detection for Older macOS

**Symptom**: Query errors on older macOS versions

**Cause**: JARVIS supports macOS Sonoma (v14) and Sequoia (v15) schemas. Older versions may have incompatible schemas.

**Solution**: Upgrade to macOS Sonoma or later.

#### 9. PDF Export Requires Additional Dependencies

**Symptom**: PDF export fails

**Cause**: PDF generation requires `weasyprint` which has system dependencies

**Solution**: Install system dependencies:
```bash
brew install pango gdk-pixbuf libffi
pip install weasyprint
```

---

## Feature Limitations

### What JARVIS Does NOT Do

1. **Send messages automatically** - Generates suggestions only
2. **Access other messaging apps** - iMessage only
3. **Work offline completely** - Model download requires internet initially
4. **Fine-tune models** - Uses RAG + few-shot (fine-tuning increases hallucinations)
5. **Store conversation history** - Reads iMessage database directly, no duplication
6. **Sync across devices** - Local-only, per-machine

### Template Coverage

The template system covers common responses (~25 templates) but won't match every message. When templates don't match:
1. System falls back to LLM generation
2. Quality depends on conversation context
3. Very specific or unusual messages may produce generic responses

### RAG Limitations

- RAG search requires embedding index build (first run)
- Cross-conversation search requires relationship registry setup
- Embedding similarity threshold (0.7) may miss relevant context

---

## Performance Characteristics

### Expected Latencies

| Operation | Cold Start | Warm Start |
|-----------|------------|------------|
| Model load | 10-15s | N/A |
| Template match | <50ms | <50ms |
| LLM generation | N/A | 2-3s |
| iMessage query | 50-200ms | 50-200ms |
| Embedding search | 100-500ms | 100-500ms |

### Memory Usage

| Component | Usage |
|-----------|-------|
| Qwen2.5-1.5B-4bit | ~1.5GB |
| Embedding model | ~400MB |
| Python runtime | ~200MB |
| Total typical | ~2.5GB |
| Peak during generation | ~4GB |

---

## Reporting New Issues

1. Check if issue is listed above
2. Run `make health` and capture output
3. Run `python -m jarvis.setup --check` and capture output
4. Include macOS version, Python version, available RAM
5. Report at: https://github.com/anthropics/claude-code/issues

---

## Workarounds Summary

| Issue | Workaround |
|-------|------------|
| Permission denied | Grant Full Disk Access |
| Model not found | `huggingface-cli download <model>` |
| Slow first generation | Expected (cold start) |
| Sending fails | Don't use sender (deprecated) |
| Wrong contact names | Check AddressBook sync |
| Group chat quality | Use simpler responses |
| PDF fails | Install weasyprint deps |
