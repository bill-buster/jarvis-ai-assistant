# Icons

To build the desktop app, you need to generate icons in the following formats:

- `icon.icns` - macOS app icon
- `icon.ico` - Windows icon
- `icon.png` - Tray icon (template)
- `32x32.png`
- `128x128.png`
- `128x128@2x.png`

## Quick Setup

1. Create a 1024x1024 PNG source icon
2. Use Tauri's icon generator:

```bash
cd desktop
npm run tauri icon path/to/your/icon.png
```

This will generate all required icon formats.
