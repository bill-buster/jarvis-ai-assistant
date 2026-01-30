# JARVIS v3 - System Flowcharts

## 1. Complete Request Flow

```mermaid
flowchart TD
    A[User clicks conversation] --> B[API receives chat_id]
    B --> C[Fetch messages from iMessage]
    C --> D[ContextAnalyzer]
    C --> E[StyleAnalyzer]
    
    D --> D1[Detect intent]
    D --> D2[Identify relationship]
    D --> D3[Summarize conversation]
    
    E --> E1[Analyze texting patterns]
    E --> E2[Check emoji usage]
    E --> E3[Identify punctuation style]
    
    D --> F[RAG Retrieval]
    E --> F
    
    F --> F1[Same-Conversation Search]
    F --> F2[Cross-Conversation Search]
    
    F1 --> F1a[Embed query]
    F1 --> F1b[FAISS search]
    F1 --> F1c[Filter by chat_id]
    F1 --> F1d[Get your replies]
    
    F2 --> F2a[Find similar contacts]
    F2 --> F2b[Search their chats]
    F2 --> F2c[Get their replies]
    
    F1 --> G[Merge & Rank Results]
    F2 --> G
    
    G --> H[Build Prompt]
    H --> H1[Add conversation context]
    H --> H2[Add style instructions]
    H --> H3[Add few-shot examples]
    
    H --> I[LFM2.5-1.2B Generation]
    I --> I1[Generate 3 replies]
    I --> I2[Apply temperature scaling]
    
    I --> J[Return to User]
```

## 2. RAG System Detail

```mermaid
flowchart LR
    subgraph "Embedding Store"
        A[Messages<br/>SQLite] --> B[Embeddings<br/>all-MiniLM]
        B --> C[FAISS Index]
    end
    
    D[Incoming Message] --> E[Embed Query]
    E --> F{Search Type}
    
    F -->|Same Chat| G[Filter by chat_id]
    F -->|Cross Chat| H[Find Similar Contacts]
    
    G --> I[Get Past Replies]
    H --> J[Search Their Chats]
    J --> K[Get Their Replies]
    
    I --> L[Merge & Rank]
    K --> L
    
    L --> M[Deduplicate]
    M --> N[Weight: Same 0.6]
    N --> O[Weight: Cross 0.4]
    O --> P[Sort by Similarity]
    P --> Q[Take Top 3-5]
    
    Q --> R[Few-Shot Examples]
```

## 3. Data Architecture

```mermaid
flowchart TB
    subgraph "Data Sources"
        A[iMessage DB<br/>chat.db]
        B[Contact Profiles<br/>contact_profiles.json]
    end
    
    subgraph "Processing"
        C[Indexer<br/>index_messages.py]
        D[Profiler<br/>profile_contacts.py]
    end
    
    subgraph "Storage"
        E[Embeddings Store<br/>SQLite + FAISS]
        F[Relationship Registry<br/>JSON]
    end
    
    subgraph "Runtime"
        G[Reply Generator]
        H[LFM2.5-1.2B Model]
    end
    
    A --> C
    B --> D
    C --> E
    D --> F
    E --> G
    F --> G
    G --> H
```

## 4. Component Interaction

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI
    participant RG as ReplyGenerator
    participant CA as ContextAnalyzer
    participant SA as StyleAnalyzer
    participant ES as EmbeddingStore
    participant RR as RelationshipRegistry
    participant ML as ModelLoader
    
    U->>API: POST /generate/replies<br/>chat_id=123
    API->>RG: generate_replies(messages, chat_id)
    
    RG->>CA: analyze(messages)
    CA-->>RG: context (intent, relationship)
    
    RG->>SA: analyze(messages, chat_id)
    SA-->>RG: style (patterns, emoji, tone)
    
    RG->>ES: find_similar_messages(query, chat_id)
    ES-->>RG: similar_msgs []
    
    RG->>RR: get_relationship_by_chat(chat_id)
    RR-->>RG: relationship_type
    
    RG->>RR: get_similar_contacts(rel_type)
    RR-->>RG: similar_contacts []
    
    RG->>ES: search_cross_conversation(query, contacts)
    ES-->>RG: cross_replies []
    
    RG->>RG: merge_results(same_replies, cross_replies)
    
    RG->>ML: generate(prompt_with_examples)
    ML-->>RG: generated_replies []
    
    RG-->>API: ReplyGenerationResult
    API-->>U: JSON response
```

## 5. Testing Strategy

```mermaid
flowchart TD
    A[Testing Strategy] --> B[Unit Tests 89%]
    A --> C[Integration 10%]
    A --> D[E2E 1%]
    
    B --> B1[test_basic.py<br/>7 tests]
    B --> B2[test_relationship_registry.py<br/>10 tests]
    B --> B3[Fast<br/>No model loading<br/>Mocked dependencies]
    
    C --> C1[API routes]
    C --> C2[Service integration]
    C --> C3[Medium speed]
    
    D --> D1[Full pipeline]
    D --> D2[Real model]
    D --> D3[Slow<br/>Validates everything]
    
    B --> E[Fast Feedback<br/><1 second]
    C --> F[Medium Feedback<br/><10 seconds]
    D --> G[Complete Validation<br/><60 seconds]
```

## 6. Roadmap Timeline

```mermaid
gantt
    title JARVIS v3 Roadmap
    dateFormat  YYYY-MM-DD
    section Phase 1
    Profile contacts           :done, p1, 2026-01-29, 2d
    Index messages             :done, p2, after p1, 1d
    Generate test samples      :active, p3, after p2, 3d
    Human evaluation           :p4, after p3, 2d
    
    section Phase 2
    Tune RAG weights           :p5, after p4, 3d
    Test same-chat weight      :p6, after p5, 1d
    Test cross-chat weight     :p7, after p6, 1d
    Test similarity threshold  :p8, after p7, 1d
    
    section Phase 3
    Prompt engineering         :p9, after p8, 3d
    Test prompt styles         :p10, after p9, 1d
    Test relationship prompts  :p11, after p10, 1d
    Test few-shot balance      :p12, after p11, 1d
    
    section Phase 4
    Evaluation framework       :p13, after p12, 3d
    Implement hybrid eval      :p14, after p13, 2d
    Calibrate with human       :p15, after p14, 2d
```

## 7. Key Decision: Why RAG Works

```mermaid
flowchart TB
    subgraph "Traditional Approach ❌"
        A1[Incoming Message] --> B1[Classify Intent]
        B1 --> C1[Generate with Intent]
        
        D1[Problem: Intent Unpredictable]
        E1["wanna hang?"]
        E1 --> F1[accept?]
        E1 --> G1[decline?]
        E1 --> H1[question?]
        E1 --> I1[reaction?]
    end
    
    subgraph "RAG Approach ✅"
        A2[Incoming Message] --> B2[Find Similar Past]
        B2 --> C2[Show Examples]
        C2 --> D2[Let Model Decide]
        
        E2["wanna hang?"]
        E2 --> F2[Find past "wanna hang?"]
        F2 --> G2[Show your replies]
        G2 --> H2["sure"]
        G2 --> I2["nah busy"]
        G2 --> J2["when?"]
    end
```

## 8. Cross-Conversation Learning

```mermaid
flowchart LR
    subgraph "Chat with Dad"
        A1[Dad: Dinner?]
        B1[You: sure]
        C1[Dad: When?]
        D1[You: 7pm]
        E1[Dad: Dinner?]
    end
    
    subgraph "Chat with Mom"
        A2[Mom: Lunch?]
        B2[You: yeah]
        C2[Mom: When?]
        D2[You: 12pm]
        E2[Mom: Dinner?]
    end
    
    E1 --> F{Same Relationship<br/>family/parent}
    E2 --> F
    
    F --> G[RAG Search]
    G --> H1["sure" (0.9)]
    G --> H2["yeah" (0.85)]
    G --> H3["7pm" (0.6)]
    
    H1 --> I[More Examples]
    H2 --> I
    H3 --> I
    I --> J[Better Replies!]
```
