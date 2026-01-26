//! JARVIS Desktop Application Entry Point
//!
//! Menu bar + window hybrid desktop app for JARVIS AI assistant.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    jarvis_desktop_lib::run()
}
