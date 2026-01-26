<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { api } from "../api/client";
  import type {
    AttachmentType,
    AttachmentWithContext,
    StorageSummary,
  } from "../api/types";

  // Props
  export let chatId: string | null = null;

  // State
  let attachments: AttachmentWithContext[] = [];
  let storageSummary: StorageSummary | null = null;
  let loading = false;
  let error: string | null = null;

  // Filters
  let selectedType: AttachmentType = "all";
  let searchQuery = "";
  let dateAfter: string = "";
  let dateBefore: string = "";

  // View mode
  let viewMode: "grid" | "list" = "grid";

  // Tab (gallery vs storage)
  let activeTab: "gallery" | "storage" = "gallery";

  // Pagination
  let limit = 100;

  // Image loading states
  let imageStates: Map<string, "loading" | "loaded" | "error"> = new Map();

  // Attachment type options
  const typeOptions: { value: AttachmentType; label: string; icon: string }[] = [
    { value: "all", label: "All", icon: "folder" },
    { value: "images", label: "Images", icon: "image" },
    { value: "videos", label: "Videos", icon: "video" },
    { value: "audio", label: "Audio", icon: "music" },
    { value: "documents", label: "Documents", icon: "file-text" },
  ];

  // Fetch attachments
  async function fetchAttachments() {
    loading = true;
    error = null;

    try {
      const options: Parameters<typeof api.getAttachments>[0] = {
        attachmentType: selectedType,
        limit,
      };

      if (chatId) {
        options.chatId = chatId;
      }
      if (dateAfter) {
        options.after = new Date(dateAfter).toISOString();
      }
      if (dateBefore) {
        options.before = new Date(dateBefore).toISOString();
      }

      attachments = await api.getAttachments(options);
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load attachments";
      attachments = [];
    } finally {
      loading = false;
    }
  }

  // Fetch storage summary
  async function fetchStorageSummary() {
    try {
      storageSummary = await api.getStorageSummary(50);
    } catch (e) {
      console.error("Failed to fetch storage summary:", e);
    }
  }

  // Filter attachments by search query
  $: filteredAttachments = searchQuery
    ? attachments.filter((a) =>
        a.attachment.filename.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : attachments;

  // Refetch when filters change
  $: if (selectedType || dateAfter || dateBefore || chatId) {
    fetchAttachments();
  }

  // Format file size
  function formatSize(bytes: number | null): string {
    if (bytes === null || bytes === 0) return "Unknown";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  }

  // Format date
  function formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    return date.toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  }

  // Get attachment type icon
  function getTypeIcon(mimeType: string | null): string {
    if (!mimeType) return "file";
    if (mimeType.startsWith("image/")) return "image";
    if (mimeType.startsWith("video/")) return "video";
    if (mimeType.startsWith("audio/")) return "music";
    if (mimeType.includes("pdf")) return "file-text";
    return "file";
  }

  // Get thumbnail URL
  function getThumbnailUrl(filePath: string | null): string | null {
    if (!filePath) return null;
    return api.getThumbnailUrl(filePath);
  }

  // Handle image load
  function handleImageLoad(filePath: string) {
    imageStates.set(filePath, "loaded");
    imageStates = imageStates;
  }

  // Handle image error
  function handleImageError(filePath: string) {
    imageStates.set(filePath, "error");
    imageStates = imageStates;
  }

  // Download attachment
  function downloadAttachment(attachment: AttachmentWithContext) {
    if (attachment.attachment.file_path) {
      window.open(api.getAttachmentUrl(attachment.attachment.file_path), "_blank");
    }
  }

  // Clear filters
  function clearFilters() {
    selectedType = "all";
    searchQuery = "";
    dateAfter = "";
    dateBefore = "";
  }

  // Calculate storage percentages
  function getStoragePercentage(bytes: number, total: number): number {
    if (total === 0) return 0;
    return (bytes / total) * 100;
  }

  // Get color for storage bar
  function getStorageColor(index: number): string {
    const colors = [
      "var(--accent-color)",
      "#34c759",
      "#ff9500",
      "#ff3b30",
      "#5856d6",
      "#007aff",
    ];
    return colors[index % colors.length];
  }

  onMount(() => {
    fetchAttachments();
    fetchStorageSummary();
  });
</script>

<div class="attachment-gallery">
  <div class="header">
    <h2>Attachments</h2>
    <div class="tabs">
      <button
        class="tab"
        class:active={activeTab === "gallery"}
        on:click={() => (activeTab = "gallery")}
      >
        Gallery
      </button>
      <button
        class="tab"
        class:active={activeTab === "storage"}
        on:click={() => (activeTab = "storage")}
      >
        Storage
      </button>
    </div>
  </div>

  {#if activeTab === "gallery"}
    <!-- Filters -->
    <div class="filters">
      <div class="filter-row">
        <div class="search-box">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="11" cy="11" r="8" />
            <path d="M21 21l-4.35-4.35" />
          </svg>
          <input
            type="text"
            placeholder="Search attachments..."
            bind:value={searchQuery}
          />
        </div>

        <div class="view-toggle">
          <button
            class:active={viewMode === "grid"}
            on:click={() => (viewMode = "grid")}
            title="Grid view"
          >
            <svg viewBox="0 0 24 24" fill="currentColor">
              <rect x="3" y="3" width="7" height="7" />
              <rect x="14" y="3" width="7" height="7" />
              <rect x="3" y="14" width="7" height="7" />
              <rect x="14" y="14" width="7" height="7" />
            </svg>
          </button>
          <button
            class:active={viewMode === "list"}
            on:click={() => (viewMode = "list")}
            title="List view"
          >
            <svg viewBox="0 0 24 24" fill="currentColor">
              <rect x="3" y="4" width="18" height="3" />
              <rect x="3" y="10.5" width="18" height="3" />
              <rect x="3" y="17" width="18" height="3" />
            </svg>
          </button>
        </div>
      </div>

      <div class="filter-row">
        <div class="type-filters">
          {#each typeOptions as option}
            <button
              class="type-filter"
              class:active={selectedType === option.value}
              on:click={() => (selectedType = option.value)}
            >
              {option.label}
            </button>
          {/each}
        </div>
      </div>

      <div class="filter-row">
        <div class="date-filters">
          <label>
            <span>After:</span>
            <input type="date" bind:value={dateAfter} />
          </label>
          <label>
            <span>Before:</span>
            <input type="date" bind:value={dateBefore} />
          </label>
          {#if dateAfter || dateBefore || selectedType !== "all" || searchQuery}
            <button class="clear-filters" on:click={clearFilters}>
              Clear filters
            </button>
          {/if}
        </div>
      </div>
    </div>

    <!-- Content -->
    {#if loading}
      <div class="loading">Loading attachments...</div>
    {:else if error}
      <div class="error">{error}</div>
    {:else if filteredAttachments.length === 0}
      <div class="empty">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
          <polyline points="14,2 14,8 20,8" />
        </svg>
        <p>No attachments found</p>
      </div>
    {:else}
      <div class="attachment-count">
        {filteredAttachments.length} attachment{filteredAttachments.length !== 1 ? "s" : ""}
      </div>

      {#if viewMode === "grid"}
        <div class="grid-view">
          {#each filteredAttachments as item}
            {@const thumbnailUrl = getThumbnailUrl(item.attachment.file_path)}
            {@const isImage = item.attachment.mime_type?.startsWith("image/")}
            {@const isVideo = item.attachment.mime_type?.startsWith("video/")}
            <div class="grid-item" on:click={() => downloadAttachment(item)}>
              <div class="thumbnail">
                {#if (isImage || isVideo) && thumbnailUrl}
                  <img
                    src={thumbnailUrl}
                    alt={item.attachment.filename}
                    on:load={() => handleImageLoad(item.attachment.file_path || "")}
                    on:error={() => handleImageError(item.attachment.file_path || "")}
                  />
                  {#if isVideo}
                    <div class="video-badge">
                      <svg viewBox="0 0 24 24" fill="currentColor">
                        <polygon points="5,3 19,12 5,21" />
                      </svg>
                    </div>
                  {/if}
                {:else}
                  <div class="file-icon">
                    {#if item.attachment.mime_type?.includes("pdf")}
                      <svg viewBox="0 0 24 24" fill="currentColor">
                        <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
                        <polyline points="14,2 14,8 20,8" fill="none" stroke="currentColor" />
                        <text x="8" y="17" font-size="6" font-weight="bold">PDF</text>
                      </svg>
                    {:else if item.attachment.mime_type?.startsWith("audio/")}
                      <svg viewBox="0 0 24 24" fill="currentColor">
                        <path d="M9 18V5l12-2v13" />
                        <circle cx="6" cy="18" r="3" />
                        <circle cx="18" cy="16" r="3" />
                      </svg>
                    {:else}
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
                        <polyline points="14,2 14,8 20,8" />
                      </svg>
                    {/if}
                  </div>
                {/if}
              </div>
              <div class="item-info">
                <span class="filename" title={item.attachment.filename}>
                  {item.attachment.filename}
                </span>
                <span class="meta">
                  {formatSize(item.attachment.file_size)}
                </span>
              </div>
            </div>
          {/each}
        </div>
      {:else}
        <div class="list-view">
          {#each filteredAttachments as item}
            <div class="list-item" on:click={() => downloadAttachment(item)}>
              <div class="list-icon">
                {#if item.attachment.mime_type?.startsWith("image/")}
                  {@const thumbnailUrl = getThumbnailUrl(item.attachment.file_path)}
                  {#if thumbnailUrl}
                    <img src={thumbnailUrl} alt="" />
                  {:else}
                    <svg viewBox="0 0 24 24" fill="currentColor">
                      <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                      <circle cx="8.5" cy="8.5" r="1.5" />
                      <polyline points="21,15 16,10 5,21" />
                    </svg>
                  {/if}
                {:else if item.attachment.mime_type?.startsWith("video/")}
                  <svg viewBox="0 0 24 24" fill="currentColor">
                    <polygon points="5,3 19,12 5,21" />
                  </svg>
                {:else if item.attachment.mime_type?.startsWith("audio/")}
                  <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M9 18V5l12-2v13" />
                    <circle cx="6" cy="18" r="3" />
                    <circle cx="18" cy="16" r="3" />
                  </svg>
                {:else}
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
                    <polyline points="14,2 14,8 20,8" />
                  </svg>
                {/if}
              </div>
              <div class="list-info">
                <span class="filename">{item.attachment.filename}</span>
                <span class="sender">
                  {item.is_from_me ? "You" : item.sender_name || item.sender}
                </span>
              </div>
              <div class="list-meta">
                <span class="size">{formatSize(item.attachment.file_size)}</span>
                <span class="date">{formatDate(item.message_date)}</span>
              </div>
            </div>
          {/each}
        </div>
      {/if}
    {/if}
  {:else}
    <!-- Storage Tab -->
    <div class="storage-view">
      {#if storageSummary}
        <div class="storage-summary">
          <div class="total-storage">
            <span class="label">Total Storage Used</span>
            <span class="value">{storageSummary.total_size_formatted}</span>
            <span class="count">{storageSummary.total_attachments} attachments</span>
          </div>
        </div>

        <div class="storage-breakdown">
          <h3>Storage by Conversation</h3>
          <div class="storage-list">
            {#each storageSummary.by_conversation as conv, i}
              <div class="storage-item">
                <div class="storage-info">
                  <span class="conv-name">{conv.display_name || "Unknown"}</span>
                  <span class="conv-meta">
                    {conv.attachment_count} files - {conv.total_size_formatted}
                  </span>
                </div>
                <div class="storage-bar-container">
                  <div
                    class="storage-bar"
                    style="width: {getStoragePercentage(
                      conv.total_size_bytes,
                      storageSummary.total_size_bytes
                    )}%; background-color: {getStorageColor(i)}"
                  ></div>
                </div>
              </div>
            {/each}
          </div>
        </div>
      {:else}
        <div class="loading">Loading storage information...</div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .attachment-gallery {
    height: 100%;
    display: flex;
    flex-direction: column;
    background: var(--bg-primary);
  }

  .header {
    padding: 16px;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .header h2 {
    font-size: 20px;
    font-weight: 600;
    margin: 0;
  }

  .tabs {
    display: flex;
    gap: 4px;
    background: var(--bg-secondary);
    padding: 4px;
    border-radius: 8px;
  }

  .tab {
    padding: 6px 12px;
    border: none;
    background: transparent;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    color: var(--text-secondary);
    transition: all 0.15s ease;
  }

  .tab:hover {
    color: var(--text-primary);
  }

  .tab.active {
    background: var(--bg-primary);
    color: var(--text-primary);
    font-weight: 500;
  }

  .filters {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .filter-row {
    display: flex;
    gap: 12px;
    align-items: center;
  }

  .search-box {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 8px 12px;
  }

  .search-box svg {
    width: 16px;
    height: 16px;
    color: var(--text-secondary);
    flex-shrink: 0;
  }

  .search-box input {
    flex: 1;
    border: none;
    background: transparent;
    color: var(--text-primary);
    font-size: 14px;
    outline: none;
  }

  .search-box input::placeholder {
    color: var(--text-secondary);
  }

  .view-toggle {
    display: flex;
    gap: 4px;
    background: var(--bg-secondary);
    padding: 4px;
    border-radius: 8px;
  }

  .view-toggle button {
    padding: 6px;
    border: none;
    background: transparent;
    border-radius: 6px;
    cursor: pointer;
    color: var(--text-secondary);
    transition: all 0.15s ease;
  }

  .view-toggle button:hover {
    color: var(--text-primary);
  }

  .view-toggle button.active {
    background: var(--bg-primary);
    color: var(--accent-color);
  }

  .view-toggle svg {
    width: 18px;
    height: 18px;
    display: block;
  }

  .type-filters {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .type-filter {
    padding: 6px 12px;
    border: 1px solid var(--border-color);
    background: var(--bg-secondary);
    border-radius: 16px;
    cursor: pointer;
    font-size: 13px;
    color: var(--text-secondary);
    transition: all 0.15s ease;
  }

  .type-filter:hover {
    border-color: var(--accent-color);
    color: var(--text-primary);
  }

  .type-filter.active {
    background: var(--accent-color);
    border-color: var(--accent-color);
    color: white;
  }

  .date-filters {
    display: flex;
    gap: 16px;
    align-items: center;
    flex-wrap: wrap;
  }

  .date-filters label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: var(--text-secondary);
  }

  .date-filters input[type="date"] {
    padding: 6px 10px;
    border: 1px solid var(--border-color);
    background: var(--bg-secondary);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
  }

  .clear-filters {
    padding: 6px 12px;
    border: none;
    background: transparent;
    color: var(--accent-color);
    cursor: pointer;
    font-size: 13px;
  }

  .clear-filters:hover {
    text-decoration: underline;
  }

  .attachment-count {
    padding: 8px 16px;
    font-size: 13px;
    color: var(--text-secondary);
    border-bottom: 1px solid var(--border-color);
  }

  .loading,
  .error,
  .empty {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: var(--text-secondary);
    gap: 12px;
  }

  .empty svg {
    width: 48px;
    height: 48px;
    opacity: 0.5;
  }

  .error {
    color: var(--error-color, #ff3b30);
  }

  /* Grid View */
  .grid-view {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 12px;
    align-content: start;
  }

  .grid-item {
    background: var(--bg-secondary);
    border-radius: 8px;
    overflow: hidden;
    cursor: pointer;
    transition: all 0.15s ease;
    border: 1px solid var(--border-color);
  }

  .grid-item:hover {
    border-color: var(--accent-color);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }

  .thumbnail {
    aspect-ratio: 1;
    background: var(--bg-tertiary, #f5f5f5);
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
  }

  .thumbnail img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .video-badge {
    position: absolute;
    bottom: 8px;
    right: 8px;
    width: 24px;
    height: 24px;
    background: rgba(0, 0, 0, 0.6);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
  }

  .video-badge svg {
    width: 12px;
    height: 12px;
  }

  .file-icon {
    width: 48px;
    height: 48px;
    color: var(--text-secondary);
  }

  .file-icon svg {
    width: 100%;
    height: 100%;
  }

  .item-info {
    padding: 8px;
  }

  .filename {
    display: block;
    font-size: 12px;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    color: var(--text-primary);
  }

  .meta {
    display: block;
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 2px;
  }

  /* List View */
  .list-view {
    flex: 1;
    overflow-y: auto;
  }

  .list-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color);
    cursor: pointer;
    transition: background 0.15s ease;
  }

  .list-item:hover {
    background: var(--bg-hover);
  }

  .list-icon {
    width: 40px;
    height: 40px;
    border-radius: 6px;
    background: var(--bg-secondary);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    flex-shrink: 0;
  }

  .list-icon img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .list-icon svg {
    width: 20px;
    height: 20px;
    color: var(--text-secondary);
  }

  .list-info {
    flex: 1;
    min-width: 0;
  }

  .list-info .filename {
    font-size: 14px;
  }

  .list-info .sender {
    display: block;
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 2px;
  }

  .list-meta {
    text-align: right;
    flex-shrink: 0;
  }

  .list-meta .size {
    display: block;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .list-meta .date {
    display: block;
    font-size: 11px;
    color: var(--text-tertiary, #999);
    margin-top: 2px;
  }

  /* Storage View */
  .storage-view {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
  }

  .storage-summary {
    margin-bottom: 24px;
  }

  .total-storage {
    background: linear-gradient(135deg, var(--accent-color), #5856d6);
    border-radius: 12px;
    padding: 24px;
    color: white;
    text-align: center;
  }

  .total-storage .label {
    display: block;
    font-size: 13px;
    opacity: 0.9;
    margin-bottom: 8px;
  }

  .total-storage .value {
    display: block;
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 4px;
  }

  .total-storage .count {
    display: block;
    font-size: 13px;
    opacity: 0.8;
  }

  .storage-breakdown h3 {
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 12px;
    color: var(--text-primary);
  }

  .storage-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .storage-item {
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 12px;
  }

  .storage-info {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
  }

  .conv-name {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 60%;
  }

  .conv-meta {
    font-size: 12px;
    color: var(--text-secondary);
    flex-shrink: 0;
  }

  .storage-bar-container {
    height: 6px;
    background: var(--bg-tertiary, #e5e5e5);
    border-radius: 3px;
    overflow: hidden;
  }

  .storage-bar {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
    min-width: 2px;
  }
</style>
