"""Contacts API endpoints.

Provides endpoints for fetching contact information including avatars.
Includes caching and default avatar generation for contacts without photos.
"""

import hashlib
import io
import logging
import threading
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from integrations.imessage import ContactAvatarData, get_contact_avatar
from integrations.imessage.parser import normalize_phone_number

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/contacts", tags=["contacts"])

# =============================================================================
# Avatar Cache with 1-hour TTL
# =============================================================================

# Cache TTL in seconds (1 hour)
AVATAR_CACHE_TTL_SECONDS = 3600.0

# Maximum cache size
AVATAR_CACHE_MAX_SIZE = 500


class AvatarCache:
    """Thread-safe avatar cache with TTL expiration.

    Stores avatar image data with 1-hour TTL to avoid repeated database queries.
    """

    def __init__(
        self, ttl_seconds: float = AVATAR_CACHE_TTL_SECONDS, maxsize: int = AVATAR_CACHE_MAX_SIZE
    ) -> None:
        """Initialize the avatar cache.

        Args:
            ttl_seconds: Time-to-live for cached items in seconds
            maxsize: Maximum number of items to cache
        """
        # Cache structure: key -> (timestamp, data)
        # data is either bytes (image) or ContactAvatarData (for generating defaults)
        self._cache: dict[str, tuple[float, Any]] = {}
        self._ttl = ttl_seconds
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> tuple[bool, Any]:
        """Get a value from the cache.

        Args:
            key: Cache key (normalized identifier)

        Returns:
            Tuple of (found, value). found is False if key doesn't exist or is expired.
        """
        with self._lock:
            if key in self._cache:
                timestamp, value = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    self._hits += 1
                    return True, value
                # Expired - remove it
                del self._cache[key]
            self._misses += 1
            return False, None

    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache.

        Args:
            key: Cache key (normalized identifier)
            value: Value to store
        """
        with self._lock:
            # Evict oldest entries if at capacity
            while len(self._cache) >= self._maxsize:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]

            self._cache[key] = (time.time(), value)

    def invalidate(self, key: str | None = None) -> None:
        """Invalidate cache entries.

        Args:
            key: Specific key to invalidate (None for all)
        """
        with self._lock:
            if key is None:
                self._cache.clear()
            elif key in self._cache:
                del self._cache[key]

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }


# Global avatar cache instance
_avatar_cache: AvatarCache | None = None
_cache_lock = threading.Lock()


def get_avatar_cache() -> AvatarCache:
    """Get the global avatar cache instance.

    Returns:
        Shared AvatarCache instance
    """
    global _avatar_cache
    if _avatar_cache is None:
        with _cache_lock:
            if _avatar_cache is None:
                _avatar_cache = AvatarCache()
    return _avatar_cache


# =============================================================================
# Default Avatar Generation
# =============================================================================

# Color palette for default avatars (pleasant, distinct colors)
AVATAR_COLORS = [
    "#FF6B6B",  # Red
    "#4ECDC4",  # Teal
    "#45B7D1",  # Sky Blue
    "#96CEB4",  # Sage
    "#FFEAA7",  # Yellow
    "#DDA0DD",  # Plum
    "#98D8C8",  # Mint
    "#F7DC6F",  # Gold
    "#BB8FCE",  # Purple
    "#85C1E9",  # Light Blue
    "#F8B500",  # Amber
    "#00CED1",  # Dark Cyan
    "#FF7F50",  # Coral
    "#9FE2BF",  # Sea Green
    "#DE3163",  # Cerise
    "#6495ED",  # Cornflower
]


def _get_color_for_identifier(identifier: str) -> str:
    """Get a consistent color for an identifier.

    Uses hash of identifier to always return the same color for
    the same contact.

    Args:
        identifier: Phone number or email

    Returns:
        Hex color string (e.g., "#FF6B6B")
    """
    hash_val = int(hashlib.md5(identifier.encode()).hexdigest(), 16)
    return AVATAR_COLORS[hash_val % len(AVATAR_COLORS)]


def _generate_initials(identifier: str, display_name: str | None = None) -> str:
    """Generate initials for an identifier.

    Args:
        identifier: Phone number or email
        display_name: Optional display name

    Returns:
        1-2 character initials string
    """
    if display_name:
        parts = display_name.split()
        if len(parts) >= 2:
            return f"{parts[0][0]}{parts[-1][0]}".upper()
        elif parts and parts[0]:
            return parts[0][0].upper()

    # For phone numbers, use last 2 digits
    if identifier.startswith("+") or identifier.isdigit():
        digits = "".join(c for c in identifier if c.isdigit())
        if len(digits) >= 2:
            return digits[-2:]
        elif digits:
            return digits[-1]

    # For emails, use first letter of local part
    if "@" in identifier:
        local_part = identifier.split("@")[0]
        if local_part:
            return local_part[0].upper()

    return "?"


def generate_svg_avatar(
    initials: str,
    background_color: str,
    size: int = 88,
) -> bytes:
    """Generate an SVG avatar with initials on a colored background.

    Args:
        initials: 1-2 character string to display
        background_color: Hex color for background (e.g., "#FF6B6B")
        size: Avatar size in pixels (default 88 for retina displays)

    Returns:
        SVG image as bytes
    """
    # Calculate font size based on number of characters
    font_size = size * 0.4 if len(initials) <= 2 else size * 0.35

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">'
        f'<circle cx="{size // 2}" cy="{size // 2}" r="{size // 2}" '
        f'fill="{background_color}"/>'
        f'<text x="50%" y="50%" dominant-baseline="central" text-anchor="middle" '
        f"font-family=\"-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif\" "
        f'font-size="{font_size}" font-weight="600" fill="white">'
        f"{initials}"
        f"</text></svg>"
    )

    return svg.encode("utf-8")


def generate_png_avatar(
    initials: str,
    background_color: str,
    size: int = 88,
) -> bytes:
    """Generate a PNG avatar with initials on a colored background.

    Falls back to SVG if PIL is not available.

    Args:
        initials: 1-2 character string to display
        background_color: Hex color for background (e.g., "#FF6B6B")
        size: Avatar size in pixels

    Returns:
        PNG image as bytes, or SVG if PIL unavailable
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        # Fall back to SVG if PIL not available
        logger.debug("PIL not available, returning SVG avatar")
        return generate_svg_avatar(initials, background_color, size)

    # Create circular avatar
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Parse hex color
    bg_color = tuple(int(background_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))

    # Draw filled circle
    draw.ellipse([0, 0, size - 1, size - 1], fill=bg_color)

    # Try to use system font, fall back to default
    font_size = int(size * 0.4) if len(initials) <= 2 else int(size * 0.35)
    try:
        # Try macOS system font
        font = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", font_size)
    except OSError:
        try:
            # Fallback to another common font
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except OSError:
            # Use default font
            font = ImageFont.load_default()  # type: ignore[assignment]

    # Calculate text position for centering
    bbox = draw.textbbox((0, 0), initials, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size - text_width) // 2
    y = (size - text_height) // 2 - bbox[1]

    # Draw text
    draw.text((x, y), initials, fill="white", font=font)

    # Convert to PNG bytes
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


# =============================================================================
# API Endpoints
# =============================================================================


@router.get(
    "/{identifier}/avatar",
    response_class=Response,
    summary="Get contact avatar",
    responses={
        200: {
            "description": "Avatar image (PNG or SVG)",
            "content": {
                "image/png": {},
                "image/svg+xml": {},
            },
        },
        400: {
            "description": "Invalid identifier format",
        },
    },
)
def get_avatar(
    identifier: str,
    size: int = Query(
        default=88,
        ge=16,
        le=512,
        description="Avatar size in pixels",
    ),
    format: str = Query(
        default="png",
        pattern="^(png|svg)$",
        description="Image format (png or svg)",
    ),
) -> Response:
    """Get avatar image for a contact.

    Returns the contact's photo from the macOS Contacts database if available.
    If no photo exists, generates a default avatar with the contact's initials
    on a colored background.

    The identifier can be:
    - Phone number (e.g., "+15551234567" or "5551234567")
    - Email address (e.g., "john@example.com")

    **Caching:**
    Avatar lookups are cached for 1 hour to minimize database queries.

    **Default Avatars:**
    When no contact photo exists, a default avatar is generated with:
    - Initials derived from contact name or identifier
    - Consistent background color based on identifier hash

    Args:
        identifier: Phone number or email address
        size: Image size in pixels (16-512, default 88)
        format: Image format - "png" or "svg" (default "png")

    Returns:
        Avatar image (PNG or SVG depending on format parameter)
    """
    # Normalize the identifier
    is_email = "@" in identifier
    normalized: str
    if is_email:
        normalized = identifier.lower().strip()
    else:
        normalized_phone = normalize_phone_number(identifier)
        if normalized_phone is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid phone number format",
            )
        normalized = normalized_phone

    # Build cache key
    cache_key = f"{normalized}:{size}:{format}"
    cache = get_avatar_cache()

    # Check cache first
    found, cached_data = cache.get(cache_key)
    if found and isinstance(cached_data, bytes):
        content_type = "image/svg+xml" if format == "svg" else "image/png"
        return Response(
            content=cached_data,
            media_type=content_type,
            headers={
                "Cache-Control": "public, max-age=3600",
                "X-Avatar-Source": "cache",
            },
        )

    # Try to get contact avatar from AddressBook
    avatar_data: ContactAvatarData | None = None
    try:
        avatar_data = get_contact_avatar(normalized)
    except Exception as e:
        logger.warning(f"Error fetching contact avatar for {normalized}: {e}")

    # If we have actual image data from contacts, use it
    if avatar_data and avatar_data.image_data:
        try:
            image_bytes = avatar_data.image_data
            # The thumbnail is typically JPEG or PNG - detect and serve as-is
            # or convert if necessary

            # Check if it's a valid image
            if image_bytes[:4] == b"\x89PNG":
                content_type = "image/png"
            elif image_bytes[:2] == b"\xff\xd8":
                content_type = "image/jpeg"
            else:
                # Try to process with PIL if available
                try:
                    from PIL import Image

                    img = Image.open(io.BytesIO(image_bytes))
                    # Resize if needed
                    if img.size[0] != size or img.size[1] != size:
                        img = img.resize((size, size), Image.Resampling.LANCZOS)  # type: ignore[assignment]
                    output = io.BytesIO()
                    img.save(output, format="PNG")
                    image_bytes = output.getvalue()
                    content_type = "image/png"
                except ImportError:
                    # Can't process, serve original
                    content_type = "image/jpeg"

            # Cache the result
            cache.set(cache_key, image_bytes)

            return Response(
                content=image_bytes,
                media_type=content_type,
                headers={
                    "Cache-Control": "public, max-age=3600",
                    "X-Avatar-Source": "contacts",
                },
            )
        except Exception as e:
            logger.warning(f"Error processing contact image: {e}")
            # Fall through to generate default

    # Generate default avatar with initials
    initials = "?"
    if avatar_data:
        initials = avatar_data.initials
    else:
        initials = _generate_initials(normalized, None)

    background_color = _get_color_for_identifier(normalized)

    if format == "svg":
        image_bytes = generate_svg_avatar(initials, background_color, size)
        content_type = "image/svg+xml"
    else:
        image_bytes = generate_png_avatar(initials, background_color, size)
        # Check if we got SVG back (PIL not available)
        if image_bytes.startswith(b"<svg"):
            content_type = "image/svg+xml"
        else:
            content_type = "image/png"

    # Cache the generated avatar
    cache.set(cache_key, image_bytes)

    return Response(
        content=image_bytes,
        media_type=content_type,
        headers={
            "Cache-Control": "public, max-age=3600",
            "X-Avatar-Source": "generated",
        },
    )


@router.get(
    "/{identifier}/info",
    summary="Get contact info",
    responses={
        200: {
            "description": "Contact information",
            "content": {
                "application/json": {
                    "example": {
                        "identifier": "+15551234567",
                        "display_name": "John Doe",
                        "has_avatar": True,
                        "initials": "JD",
                        "avatar_color": "#FF6B6B",
                    }
                }
            },
        },
        400: {
            "description": "Invalid identifier format",
        },
    },
)
def get_contact_info(identifier: str) -> dict[str, Any]:
    """Get basic contact information.

    Returns contact name and avatar metadata without the actual image data.
    Useful for efficiently loading contact info for multiple contacts.

    Args:
        identifier: Phone number or email address

    Returns:
        JSON object with contact information
    """
    # Normalize the identifier
    is_email = "@" in identifier
    normalized: str
    if is_email:
        normalized = identifier.lower().strip()
    else:
        normalized_phone = normalize_phone_number(identifier)
        if normalized_phone is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid phone number format",
            )
        normalized = normalized_phone

    # Try to get contact info
    avatar_data: ContactAvatarData | None = None
    try:
        avatar_data = get_contact_avatar(normalized)
    except Exception as e:
        logger.warning(f"Error fetching contact info for {normalized}: {e}")

    display_name = None
    has_avatar = False
    initials = _generate_initials(normalized, None)

    if avatar_data:
        has_avatar = avatar_data.image_data is not None
        initials = avatar_data.initials
        if avatar_data.display_name:
            display_name = avatar_data.display_name
        elif avatar_data.first_name or avatar_data.last_name:
            parts = []
            if avatar_data.first_name:
                parts.append(avatar_data.first_name)
            if avatar_data.last_name:
                parts.append(avatar_data.last_name)
            display_name = " ".join(parts)

    return {
        "identifier": normalized,
        "display_name": display_name,
        "has_avatar": has_avatar,
        "initials": initials,
        "avatar_color": _get_color_for_identifier(normalized),
    }
