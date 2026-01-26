//! JARVIS Desktop Application Library
//!
//! Provides the Tauri application setup and tray functionality.

mod tray;

use tauri::Manager;

/// Run the Tauri application
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // Set up the system tray
            tray::setup_tray(app)?;

            // Get the main window
            if let Some(window) = app.get_webview_window("main") {
                // Handle close button - hide instead of quit
                let window_clone = window.clone();
                window.on_window_event(move |event| {
                    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                        // Hide the window instead of closing
                        let _ = window_clone.hide();
                        api.prevent_close();
                    }
                });
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
