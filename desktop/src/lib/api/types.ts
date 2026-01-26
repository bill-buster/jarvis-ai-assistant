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

// Quality Metrics types
export interface QualitySummary {
  total_responses: number;
  template_responses: number;
  model_responses: number;
  template_hit_rate_percent: number;
  model_fallback_rate_percent: number;
  avg_hhem_score: number | null;
  hhem_score_count: number;
  acceptance_rate_percent: number;
  accepted_unchanged_count: number;
  accepted_modified_count: number;
  rejected_count: number;
  avg_edit_distance: number | null;
  avg_template_latency_ms: number;
  avg_model_latency_ms: number;
  uptime_seconds: number;
  responses_per_second: number;
}

export interface TrendDataPoint {
  date: string;
  template_hit_rate_percent: number;
  model_hit_rate_percent: number;
  avg_hhem_score: number | null;
  acceptance_rate_percent: number;
  avg_edit_distance: number | null;
  avg_template_latency_ms: number;
  avg_model_latency_ms: number;
  total_responses: number;
}

export interface ContactQuality {
  contact_id: string;
  total_responses: number;
  template_responses: number;
  model_responses: number;
  acceptance_rate: number;
  avg_hhem_score: number | null;
  avg_edit_distance: number | null;
  avg_latency_ms: number;
}

export interface TimeOfDayQuality {
  hour: number;
  total_responses: number;
  template_hit_rate: number;
  acceptance_rate: number;
  avg_latency_ms: number;
}

export interface IntentQuality {
  intent: string;
  total_responses: number;
  template_hit_rate: number;
  acceptance_rate: number;
  avg_hhem_score: number | null;
  avg_latency_ms: number;
}

export interface ConversationTypeQuality {
  total_responses: number;
  template_hit_rate_percent: number;
  acceptance_rate_percent: number;
  avg_hhem_score: number | null;
  avg_latency_ms: number;
}

export interface Recommendation {
  category: string;
  priority: string;
  title: string;
  description: string;
  metric_value: number | null;
  target_value: number | null;
}

export interface QualityDashboardData {
  summary: QualitySummary;
  trends: TrendDataPoint[];
  top_contacts: ContactQuality[];
  time_of_day: TimeOfDayQuality[];
  by_intent: IntentQuality[];
  by_conversation_type: Record<string, ConversationTypeQuality>;
  recommendations: Recommendation[];
}
