    calendar_id: str = Field(
        ...,
        description="Target calendar ID",
    )
    detected_event: DetectedEventResponse = Field(
        ...,
        description="Detected event to add to calendar",
    )


class CreateEventResponse(BaseModel):
    """Response after creating a calendar event.

    Example:
        ```json
        {
            "success": true,
            "event_id": "event-789",
            "error": null
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "event_id": "event-789",
                "error": None,
            }
        }
    )

    success: bool = Field(
        ...,
        description="Whether the event was created successfully",
    )
    event_id: str | None = Field(
        default=None,
        description="ID of created event (if successful)",
    )
    error: str | None = Field(
        default=None,
        description="Error message (if failed)",
=======
    imported: int = Field(
        ...,
        ge=0,
        description="Number of templates successfully imported",
    )
    skipped: int = Field(
        ...,
        ge=0,
        description="Number of templates skipped",
    )
    errors: int = Field(
        ...,
        ge=0,
        description="Number of templates that failed to import",
    )
    total_templates: int = Field(
        ...,
        ge=0,
        description="Total templates after import",
>>>>>>> origin/claude/custom-template-builder-VPeOl
    )
