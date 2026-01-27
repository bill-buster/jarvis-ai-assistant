/**
 * API type definitions for JARVIS frontend
 */

// Message types
export interface Attachment {
  filename: string;
  file_path: string | null;
  mime_type: string | null;
  file_size: number | null;
}

export interface Reaction {
  type: string;
  sender: string;
  sender_name: string | null;
  date: string;
}

export interface Message {
  id: number;
  chat_id: string;
  sender: string;
  sender_name: string | null;
  text: string;
  date: string;
  is_from_me: boolean;
  attachments: Attachment[];
  reply_to_id: number | null;
  reactions: Reaction[];
  date_delivered: string | null;
  date_read: string | null;
  is_system_message: boolean;
}

// Conversation types
export interface Conversation {
  chat_id: string;
  participants: string[];
  display_name: string | null;
  last_message_date: string;
  message_count: number;
  is_group: boolean;
  last_message_text: string | null;
}

// Health types
export interface HealthResponse {
  status: "healthy" | "degraded" | "unhealthy";
  imessage_access: boolean;
  memory_available_gb: number;
  memory_used_gb: number;
  memory_mode: "FULL" | "LITE" | "MINIMAL";
  model_loaded: boolean;
  permissions_ok: boolean;
  details: Record<string, string> | null;
  jarvis_rss_mb: number;
  jarvis_vms_mb: number;
}

// Settings types
export interface ModelInfo {
  model_id: string;
  name: string;
  size_gb: number;
  quality_tier: "basic" | "good" | "best";
  ram_requirement_gb: number;
  is_downloaded: boolean;
  is_loaded: boolean;
  is_recommended: boolean;
  description: string | null;
}

export interface GenerationSettings {
  temperature: number;
  max_tokens_reply: number;
  max_tokens_summary: number;
}

export interface BehaviorSettings {
  auto_suggest_replies: boolean;
  suggestion_count: number;
  context_messages_reply: number;
  context_messages_summary: number;
}

export interface SystemInfo {
  system_ram_gb: number;
  current_memory_usage_gb: number;
  model_loaded: boolean;
  model_memory_usage_gb: number;
  imessage_access: boolean;
}

export interface SettingsResponse {
  model_id: string;
  generation: GenerationSettings;
  behavior: BehaviorSettings;
  system: SystemInfo;
}

export interface SettingsUpdateRequest {
  model_id?: string;
  generation?: Partial<GenerationSettings>;
  behavior?: Partial<BehaviorSettings>;
}

export interface DownloadStatus {
  model_id: string;
  status: "downloading" | "completed" | "failed";
  progress: number;
  error: string | null;
}

export interface ActivateResponse {
  success: boolean;
  model_id: string;
  error: string | null;
}

// Draft and Summary types
export interface DraftSuggestion {
  text: string;
  confidence: number;
}

export interface ContextInfo {
  num_messages: number;
  participants: string[];
  last_message: string | null;
}

export interface DraftReplyResponse {
  suggestions: DraftSuggestion[];
  context_used: ContextInfo;
}

export interface DateRange {
  start: string;
  end: string;
}

export interface SummaryResponse {
  summary: string;
  key_points: string[];
  date_range: DateRange;
  message_count?: number;
}

// Smart Reply Suggestion types
export interface SmartReplySuggestion {
  text: string;
  score: number;
}

export interface SmartReplySuggestionsResponse {
  suggestions: SmartReplySuggestion[];
}

// Topic types
export interface Topic {
  topic: string;
  confidence: number;
  color: string;
  display_name: string;
}

export interface TopicsResponse {
  chat_id: string;
  topics: Topic[];
  all_topics: Topic[];
  cached: boolean;
  message_count_analyzed: number;
}

// Search types
export interface SearchFilters {
  sender?: string;
  after?: string;
  before?: string;
  has_attachments?: boolean;
}

export interface SearchResult extends Message {
  conversation_name?: string;
}

export interface SearchResponse {
  results: SearchResult[];
  query: string;
  filters_applied: SearchFilters;
}

// Grouped search results by conversation
export interface GroupedSearchResults {
  chat_id: string;
  conversation_name: string;
  results: SearchResult[];
}

// Error types
export interface ApiError {
  error: string;
  detail: string;
  code: string | null;
}

// Statistics types
export type TimeRange = "week" | "month" | "three_months" | "all_time";

export interface HourlyActivity {
  hour: number;
  count: number;
}

export interface WordFrequency {
  word: string;
  count: number;
}

export interface ConversationStats {
  chat_id: string;
  time_range: TimeRange;
  total_messages: number;
  sent_count: number;
  received_count: number;
  avg_response_time_minutes: number | null;
  hourly_activity: HourlyActivity[];
  daily_activity: Record<string, number>;
  message_length_distribution: Record<string, number>;
  top_words: WordFrequency[];
  emoji_usage: Record<string, number>;
  attachment_breakdown: Record<string, number>;
  first_message_date: string | null;
  last_message_date: string | null;
  participants: string[];
}

// Template Analytics types
export interface TemplateAnalyticsSummary {
  total_queries: number;
  template_hits: number;
  model_fallbacks: number;
  hit_rate_percent: number;
  cache_hit_rate: number;
  unique_templates_matched: number;
  queries_per_second: number;
  uptime_seconds: number;
}

export interface TopTemplateItem {
  template_name: string;
  match_count: number;
}

export interface MissedQueryItem {
  query_hash: string;
  similarity: number;
  best_template: string | null;
  timestamp: string;
}

export interface CategoryAverageItem {
  category: string;
  average_similarity: number;
}

export interface TemplateCoverage {
  total_templates: number;
  total_patterns: number;
  responses_from_templates: number;
  responses_from_model: number;
  coverage_percent: number;
}

export interface PieChartData {
  template_responses: number;
  model_responses: number;
}

export interface TemplateAnalyticsDashboard {
  summary: TemplateAnalyticsSummary;
  top_templates: TopTemplateItem[];
  missed_queries: MissedQueryItem[];
  category_averages: CategoryAverageItem[];
  coverage: TemplateCoverage;
  pie_chart_data: PieChartData;
}

// Semantic Search types
export interface SemanticSearchFilters {
  sender?: string;
  chat_id?: string;
  after?: string;
  before?: string;
  has_attachments?: boolean;
}

export interface SemanticSearchRequest {
  query: string;
  limit?: number;
  threshold?: number;
  index_limit?: number;
  filters?: SemanticSearchFilters;
}

export interface SemanticSearchResultItem {
  message: Message;
  similarity: number;
}

export interface SemanticSearchResponse {
  query: string;
  results: SemanticSearchResultItem[];
  total_results: number;
  threshold_used: number;
  messages_searched: number;
}

export interface SemanticCacheStats {
  embedding_count: number;
  size_bytes: number;
  size_mb: number;
}

// Priority Inbox types
export type PriorityLevel = "critical" | "high" | "medium" | "low";

export type PriorityReason =
  | "contains_question"
  | "action_requested"
  | "time_sensitive"
  | "important_contact"
  | "frequent_contact"
  | "awaiting_response"
  | "multiple_messages"
  | "contains_urgency"
  | "normal";

export interface PriorityMessage {
  message_id: number;
  chat_id: string;
  sender: string;
  sender_name: string | null;
  text: string;
  date: string;
  priority_score: number;
  priority_level: PriorityLevel;
  reasons: PriorityReason[];
  needs_response: boolean;
  handled: boolean;
  conversation_name: string | null;
}

export interface PriorityInboxResponse {
  messages: PriorityMessage[];
  total_count: number;
  unhandled_count: number;
  needs_response_count: number;
  critical_count: number;
  high_count: number;
}

export interface MarkHandledRequest {
  chat_id: string;
  message_id: number;
}

export interface MarkHandledResponse {
  success: boolean;
  chat_id: string;
  message_id: number;
  handled: boolean;
}

export interface ImportantContactRequest {
  identifier: string;
  important: boolean;
}

export interface ImportantContactResponse {
  success: boolean;
  identifier: string;
  important: boolean;
}

export interface PriorityStats {
  handled_count: number;
  important_contacts_count: number;
}

// PDF Export types
export interface PDFExportDateRange {
  start?: string;
  end?: string;
}

export interface PDFExportRequest {
  include_attachments?: boolean;
  include_reactions?: boolean;
  date_range?: PDFExportDateRange;
  limit?: number;
}

export interface PDFExportResponse {
  success: boolean;
  filename: string;
  data: string; // Base64-encoded PDF
  message_count: number;
  size_bytes: number;
}

// WebSocket types
export interface WebSocketStatus {
  active_connections: number;
  health_subscribers: number;
  status: "operational" | "degraded" | "offline";
}

export interface GenerationStreamRequest {
  prompt: string;
  context_documents?: string[];
  few_shot_examples?: Array<{ input: string; output: string }>;
  max_tokens?: number;
  temperature?: number;
  stop_sequences?: string[];
}

export interface StreamingGenerationResult {
  generation_id: string;
  text: string;
  tokens_used: number;
  generation_time_ms: number;
  model_name: string;
  used_template: boolean;
  template_name: string | null;
  finish_reason: string;
}

// Insights types
export interface SentimentResult {
  score: number;
  label: "positive" | "negative" | "neutral";
  positive_count: number;
  negative_count: number;
  neutral_count: number;
}

export interface SentimentTrend {
  date: string;
  score: number;
  message_count: number;
}

export interface ResponsePatterns {
  avg_response_time_minutes: number | null;
  median_response_time_minutes: number | null;
  fastest_response_minutes: number | null;
  slowest_response_minutes: number | null;
  response_times_by_hour: Record<number, number>;
  response_times_by_day: Record<string, number>;
  my_avg_response_time_minutes: number | null;
  their_avg_response_time_minutes: number | null;
}

export interface FrequencyTrends {
  daily_counts: Record<string, number>;
  weekly_counts: Record<string, number>;
  monthly_counts: Record<string, number>;
  trend_direction: "increasing" | "decreasing" | "stable";
  trend_percentage: number;
  most_active_day: string | null;
  most_active_hour: number | null;
  messages_per_day_avg: number;
}

export interface RelationshipHealth {
  overall_score: number;
  engagement_score: number;
  sentiment_score: number;
  responsiveness_score: number;
  consistency_score: number;
  health_label: "excellent" | "good" | "fair" | "needs_attention" | "concerning";
  factors: Record<string, string>;
}

export interface ConversationInsights {
  chat_id: string;
  contact_name: string | null;
  time_range: TimeRange;
  sentiment_overall: SentimentResult;
  sentiment_trends: SentimentTrend[];
  response_patterns: ResponsePatterns;
  frequency_trends: FrequencyTrends;
  relationship_health: RelationshipHealth;
  total_messages_analyzed: number;
  first_message_date: string | null;
  last_message_date: string | null;
}

// Calendar types
export interface DetectedEvent {
  title: string;
  start: string;
  end: string | null;
  location: string | null;
  description: string | null;
  all_day: boolean;
  confidence: number;
  source_text: string;
  message_id: number | null;
}

export interface Calendar {
  id: string;
  name: string;
  color: string | null;
  is_editable: boolean;
}

export interface CalendarEvent {
  id: string;
  calendar_id: string;
  calendar_name: string;
  title: string;
  start: string;
  end: string;
  all_day: boolean;
  location: string | null;
  notes: string | null;
  url: string | null;
  attendees: string[];
  status: "confirmed" | "tentative" | "cancelled";
}

export interface CreateEventResponse {
  success: boolean;
  event_id: string | null;
  error: string | null;
}

// Custom Template types
export interface CustomTemplate {
  id: string;
  name: string;
  template_text: string;
  trigger_phrases: string[];
  category: string;
  tags: string[];
  min_group_size: number | null;
  max_group_size: number | null;
  enabled: boolean;
  created_at: string;
  updated_at: string;
  usage_count: number;
}

export interface CustomTemplateCreateRequest {
  name: string;
  template_text: string;
  trigger_phrases: string[];
  category?: string;
  tags?: string[];
  min_group_size?: number | null;
  max_group_size?: number | null;
  enabled?: boolean;
}

export interface CustomTemplateUpdateRequest {
  name?: string;
  template_text?: string;
  trigger_phrases?: string[];
  category?: string;
  tags?: string[];
  min_group_size?: number | null;
  max_group_size?: number | null;
  enabled?: boolean;
}

export interface CustomTemplateListResponse {
  templates: CustomTemplate[];
  total: number;
  categories: string[];
  tags: string[];
}

export interface CustomTemplateUsageStats {
  total_templates: number;
  enabled_templates: number;
  total_usage: number;
  usage_by_category: Record<string, number>;
  top_templates: Array<{ id: string; name: string; usage_count: number }>;
}

export interface CustomTemplateTestRequest {
  trigger_phrases: string[];
  test_inputs: string[];
}

export interface CustomTemplateTestResult {
  input: string;
  matched: boolean;
  best_match: string | null;
  similarity: number;
}

export interface CustomTemplateTestResponse {
  results: CustomTemplateTestResult[];
  match_rate: number;
  threshold: number;
}

export interface CustomTemplateExportResponse {
  version: number;
  export_date: string;
  template_count: number;
  templates: Record<string, unknown>[];
}

export interface CustomTemplateImportRequest {
  data: Record<string, unknown>;
  overwrite?: boolean;
}

export interface CustomTemplateImportResponse {
  imported: number;
  skipped: number;
  errors: number;
  total_templates: number;
}

// Threading types
export interface ThreadResponse {
  thread_id: string;
  messages: number[];
  topic_label: string;
  start_time: string | null;
  end_time: string | null;
  participant_count: number;
  message_count: number;
}

export interface ThreadedMessage extends Message {
  thread_id: string;
  thread_position: number;
  is_thread_start: boolean;
}

export interface ThreadedViewResponse {
  chat_id: string;
  threads: ThreadResponse[];
  messages: ThreadedMessage[];
  total_threads: number;
  total_messages: number;
}

export interface ThreadingConfigRequest {
  time_gap_threshold_minutes?: number;
  semantic_similarity_threshold?: number;
  use_semantic_analysis?: boolean;
}

// Attachment Manager types
export type AttachmentType = "images" | "videos" | "audio" | "documents" | "other" | "all";

export interface ExtendedAttachment {
  filename: string;
  file_path: string | null;
  mime_type: string | null;
  file_size: number | null;
  width: number | null;
  height: number | null;
  duration_seconds: number | null;
  created_date: string | null;
  is_sticker: boolean;
  uti: string | null;
}

export interface AttachmentWithContext {
  attachment: ExtendedAttachment;
  message_id: number;
  message_date: string;
  chat_id: string;
  sender: string;
  sender_name: string | null;
  is_from_me: boolean;
}

export interface AttachmentStats {
  chat_id: string;
  total_count: number;
  total_size_bytes: number;
  total_size_formatted: string;
  by_type: Record<string, number>;
  size_by_type: Record<string, number>;
}

export interface StorageByConversation {
  chat_id: string;
  display_name: string | null;
  attachment_count: number;
  total_size_bytes: number;
  total_size_formatted: string;
}

export interface StorageSummary {
  total_attachments: number;
  total_size_bytes: number;
  total_size_formatted: string;
  by_conversation: StorageByConversation[];
}
