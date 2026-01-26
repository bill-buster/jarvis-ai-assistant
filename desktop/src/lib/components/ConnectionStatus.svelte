<script lang="ts">
  /**
   * Connection status indicator for WebSocket connectivity
   *
   * Displays the current WebSocket connection state with visual feedback.
   */
  import { connectionState, isConnected } from "../stores/websocket";

  // Map connection states to display info
  const stateInfo: Record<
    string,
    { label: string; color: string; pulse: boolean }
  > = {
    connected: { label: "Connected", color: "#34c759", pulse: false },
    connecting: { label: "Connecting", color: "#ff9f0a", pulse: true },
    reconnecting: { label: "Reconnecting", color: "#ff9f0a", pulse: true },
    disconnected: { label: "Disconnected", color: "#ff5f57", pulse: false },
  };

  $: info = stateInfo[$connectionState] || stateInfo.disconnected;
</script>

<div class="connection-status" title="WebSocket connection status">
  <div
    class="status-dot"
    class:pulse={info.pulse}
    style="background-color: {info.color}"
  ></div>
  <span class="status-label">{info.label}</span>
</div>

<style>
  .connection-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    font-size: 12px;
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .status-dot.pulse {
    animation: pulse 1.5s ease-in-out infinite;
  }

  @keyframes pulse {
    0%,
    100% {
      opacity: 1;
      transform: scale(1);
    }
    50% {
      opacity: 0.5;
      transform: scale(1.2);
    }
  }

  .status-label {
    color: var(--text-secondary);
    white-space: nowrap;
  }
</style>
