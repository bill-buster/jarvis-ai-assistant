"""Custom Templates API endpoints.

Provides CRUD operations for user-defined templates, including:
- Create, read, update, delete templates
- Test templates against sample inputs
- Import/export template packs for sharing
- Usage statistics and analytics
"""

import logging

import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas import (
    CustomTemplateCreateRequest,
    CustomTemplateExportRequest,
    CustomTemplateExportResponse,
    CustomTemplateImportRequest,
    CustomTemplateImportResponse,
    CustomTemplateListResponse,
    CustomTemplateResponse,
    CustomTemplateTestRequest,
    CustomTemplateTestResponse,
    CustomTemplateTestResult,
    CustomTemplateUpdateRequest,
    CustomTemplateUsageStats,
    ErrorResponse,
)
from models.templates import (
    CustomTemplate,
    TemplateMatcher,
    get_custom_template_store,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/templates", tags=["custom-templates"])


def _template_to_response(template: CustomTemplate) -> CustomTemplateResponse:
    """Convert a CustomTemplate to CustomTemplateResponse."""
    return CustomTemplateResponse(
        id=template.id,
        name=template.name,
        template_text=template.template_text,
        trigger_phrases=template.trigger_phrases,
        category=template.category,
        tags=template.tags,
        min_group_size=template.min_group_size,
        max_group_size=template.max_group_size,
        enabled=template.enabled,
        created_at=template.created_at,
        updated_at=template.updated_at,
        usage_count=template.usage_count,
    )


@router.get(
    "",
    response_model=CustomTemplateListResponse,
    response_model_exclude_unset=True,
    response_description="List of custom templates with metadata",
    summary="List all custom templates",
    responses={
        200: {
            "description": "Templates retrieved successfully",
        }
    },
)
def list_templates(
    category: str | None = None,
    tag: str | None = None,
    enabled_only: bool = False,
) -> CustomTemplateListResponse:
    """List all custom templates with optional filtering.

    Query Parameters:
    - **category**: Filter by category
    - **tag**: Filter by tag
    - **enabled_only**: Only return enabled templates

    Returns:
        CustomTemplateListResponse with templates and metadata
    """
    store = get_custom_template_store()

    # Get templates with filters
    if category:
        templates = store.list_by_category(category)
    elif tag:
        templates = store.list_by_tag(tag)
    elif enabled_only:
        templates = store.list_enabled()
    else:
        templates = store.list_all()

    return CustomTemplateListResponse(
        templates=[_template_to_response(t) for t in templates],
        total=len(templates),
        categories=store.get_categories(),
        tags=store.get_tags(),
    )


@router.post(
    "",
    response_model=CustomTemplateResponse,
    response_model_exclude_unset=True,
    response_description="The created template",
    summary="Create a new custom template",
    status_code=201,
    responses={
        201: {
            "description": "Template created successfully",
        },
        400: {
            "description": "Invalid template data",
            "model": ErrorResponse,
        },
    },
)
def create_template(request: CustomTemplateCreateRequest) -> CustomTemplateResponse:
    """Create a new custom template.

    **Required Fields:**
    - name: Human-readable template name
    - template_text: The response to return when matched
    - trigger_phrases: At least one phrase to trigger this template

    **Optional Fields:**
    - category: Category for organization (default: "general")
    - tags: Additional tags for filtering
    - min_group_size/max_group_size: Group size constraints
    - enabled: Whether the template is active (default: True)

    Returns:
        The created CustomTemplateResponse with ID assigned
    """
    store = get_custom_template_store()

    # Validate trigger phrases
    if not request.trigger_phrases:
        raise HTTPException(
            status_code=400,
            detail="At least one trigger phrase is required",
        )

    # Validate group size constraints
    if (
        request.min_group_size is not None
        and request.max_group_size is not None
        and request.min_group_size > request.max_group_size
    ):
        raise HTTPException(
            status_code=400,
            detail="min_group_size cannot be greater than max_group_size",
        )

    # Create template
    template = CustomTemplate(
        name=request.name,
        template_text=request.template_text,
        trigger_phrases=request.trigger_phrases,
        category=request.category,
        tags=request.tags,
        min_group_size=request.min_group_size,
        max_group_size=request.max_group_size,
        enabled=request.enabled,
    )

    created = store.create(template)
    logger.info("Created custom template: %s (id=%s)", created.name, created.id)

    return _template_to_response(created)


@router.get(
    "/{template_id}",
    response_model=CustomTemplateResponse,
    response_model_exclude_unset=True,
    response_description="The requested template",
    summary="Get a specific template",
    responses={
        200: {
            "description": "Template retrieved successfully",
        },
        404: {
            "description": "Template not found",
            "model": ErrorResponse,
        },
    },
)
def get_template(template_id: str) -> CustomTemplateResponse:
    """Get a specific custom template by ID.

    Args:
        template_id: The template UUID

    Returns:
        The CustomTemplateResponse

    Raises:
        HTTPException 404: Template not found
    """
    store = get_custom_template_store()
    template = store.get(template_id)

    if template is None:
        raise HTTPException(
            status_code=404,
            detail=f"Template not found: {template_id}",
        )

    return _template_to_response(template)


@router.put(
    "/{template_id}",
    response_model=CustomTemplateResponse,
    response_model_exclude_unset=True,
    response_description="The updated template",
    summary="Update a custom template",
    responses={
        200: {
            "description": "Template updated successfully",
        },
        400: {
            "description": "Invalid update data",
            "model": ErrorResponse,
        },
        404: {
            "description": "Template not found",
            "model": ErrorResponse,
        },
    },
)
def update_template(
    template_id: str, request: CustomTemplateUpdateRequest
) -> CustomTemplateResponse:
    """Update an existing custom template.

    Performs a partial update - only provided fields are changed.

    Args:
        template_id: The template UUID to update
        request: Fields to update

    Returns:
        The updated CustomTemplateResponse

    Raises:
        HTTPException 404: Template not found
        HTTPException 400: Invalid update data
    """
    store = get_custom_template_store()

    # Check if template exists
    existing = store.get(template_id)
    if existing is None:
        raise HTTPException(
            status_code=404,
            detail=f"Template not found: {template_id}",
        )

    # Build updates dict from provided fields
    updates: dict[str, str | list[str] | int | bool | None] = {}
    if request.name is not None:
        updates["name"] = request.name
    if request.template_text is not None:
        updates["template_text"] = request.template_text
    if request.trigger_phrases is not None:
        if not request.trigger_phrases:
            raise HTTPException(
                status_code=400,
                detail="At least one trigger phrase is required",
            )
        updates["trigger_phrases"] = request.trigger_phrases
    if request.category is not None:
        updates["category"] = request.category
    if request.tags is not None:
        updates["tags"] = request.tags
    if request.min_group_size is not None:
        updates["min_group_size"] = request.min_group_size
    if request.max_group_size is not None:
        updates["max_group_size"] = request.max_group_size
    if request.enabled is not None:
        updates["enabled"] = request.enabled

    # Validate group size constraints
    min_size_val = updates.get("min_group_size", existing.min_group_size)
    max_size_val = updates.get("max_group_size", existing.max_group_size)

    # Validate types explicitly
    if min_size_val is not None and not isinstance(min_size_val, int):
        raise HTTPException(
            status_code=400,
            detail="min_group_size must be an integer",
        )
    if max_size_val is not None and not isinstance(max_size_val, int):
        raise HTTPException(
            status_code=400,
            detail="max_group_size must be an integer",
        )

    min_size = min_size_val
    max_size = max_size_val
    if min_size is not None and max_size is not None and min_size > max_size:
        raise HTTPException(
            status_code=400,
            detail="min_group_size cannot be greater than max_group_size",
        )

    updated = store.update(template_id, updates)
    if updated is None:
        raise HTTPException(
            status_code=404,
            detail=f"Template not found: {template_id}",
        )

    logger.info("Updated custom template: %s (id=%s)", updated.name, updated.id)
    return _template_to_response(updated)


@router.delete(
    "/{template_id}",
    response_model=dict,
    response_description="Deletion confirmation",
    summary="Delete a custom template",
    responses={
        200: {
            "description": "Template deleted successfully",
            "content": {
                "application/json": {"example": {"status": "deleted", "template_id": "abc123"}}
            },
        },
        404: {
            "description": "Template not found",
            "model": ErrorResponse,
        },
    },
)
def delete_template(template_id: str) -> dict[str, str]:
    """Delete a custom template.

    Args:
        template_id: The template UUID to delete

    Returns:
        Confirmation with deleted template ID

    Raises:
        HTTPException 404: Template not found
    """
    store = get_custom_template_store()

    if not store.delete(template_id):
        raise HTTPException(
            status_code=404,
            detail=f"Template not found: {template_id}",
        )

    logger.info("Deleted custom template: %s", template_id)
    return {"status": "deleted", "template_id": template_id}


@router.get(
    "/stats/usage",
    response_model=CustomTemplateUsageStats,
    response_model_exclude_unset=True,
    response_description="Template usage statistics",
    summary="Get template usage statistics",
    responses={
        200: {
            "description": "Statistics retrieved successfully",
        }
    },
)
def get_usage_stats() -> CustomTemplateUsageStats:
    """Get usage statistics for all custom templates.

    Returns:
        CustomTemplateUsageStats with counts and top templates
    """
    store = get_custom_template_store()
    stats = store.get_usage_stats()

    return CustomTemplateUsageStats(
        total_templates=stats["total_templates"],
        enabled_templates=stats["enabled_templates"],
        total_usage=stats["total_usage"],
        usage_by_category=stats["usage_by_category"],
        top_templates=stats["top_templates"],
    )


@router.post(
    "/test",
    response_model=CustomTemplateTestResponse,
    response_model_exclude_unset=True,
    response_description="Test results for template matching",
    summary="Test template against sample inputs",
    responses={
        200: {
            "description": "Test completed successfully",
        },
        400: {
            "description": "Invalid test request",
            "model": ErrorResponse,
        },
        503: {
            "description": "Sentence model unavailable",
            "model": ErrorResponse,
        },
    },
)
def test_template(request: CustomTemplateTestRequest) -> CustomTemplateTestResponse:
    """Test template trigger phrases against sample inputs.

    This endpoint helps validate that trigger phrases will match
    the intended inputs before saving a template.

    Args:
        request: Trigger phrases and test inputs

    Returns:
        CustomTemplateTestResponse with match results for each input
    """
    if not request.trigger_phrases:
        raise HTTPException(
            status_code=400,
            detail="At least one trigger phrase is required",
        )

    if not request.test_inputs:
        raise HTTPException(
            status_code=400,
            detail="At least one test input is required",
        )

    try:
        # Create a temporary template matcher with just the trigger phrases
        from models.templates import ResponseTemplate

        temp_template = ResponseTemplate(
            name="test_template",
            patterns=request.trigger_phrases,
            response="test response",
        )
        matcher = TemplateMatcher(templates=[temp_template])

        results: list[CustomTemplateTestResult] = []
        matches = 0

        for test_input in request.test_inputs:
            match_result = matcher.match(test_input, track_analytics=False)

            if match_result is not None:
                results.append(
                    CustomTemplateTestResult(
                        input=test_input,
                        matched=True,
                        best_match=match_result.matched_pattern,
                        similarity=match_result.similarity,
                    )
                )
                matches += 1
            else:
                # Get best similarity even if below threshold
                matcher._ensure_embeddings()
                pattern_embeddings = matcher._pattern_embeddings
                if pattern_embeddings is not None:
                    query_embedding = matcher._get_query_embedding(test_input)
                    query_norm = np.linalg.norm(query_embedding)
                    pattern_norms = matcher._pattern_norms
                    if pattern_norms is None:
                        pattern_norms = np.linalg.norm(pattern_embeddings, axis=1)
                        pattern_norms = np.where(pattern_norms == 0, 1, pattern_norms)

                    dot_products = np.dot(pattern_embeddings, query_embedding)
                    similarities = dot_products / (pattern_norms * query_norm)
                    best_idx = int(np.argmax(similarities))
                    best_similarity = float(similarities[best_idx])
                    best_pattern = matcher._pattern_to_template[best_idx][0]

                    results.append(
                        CustomTemplateTestResult(
                            input=test_input,
                            matched=False,
                            best_match=best_pattern,
                            similarity=best_similarity,
                        )
                    )
                else:
                    results.append(
                        CustomTemplateTestResult(
                            input=test_input,
                            matched=False,
                            best_match=None,
                            similarity=0.0,
                        )
                    )

        match_rate = matches / len(request.test_inputs) if request.test_inputs else 0.0

        return CustomTemplateTestResponse(
            results=results,
            match_rate=match_rate,
            threshold=TemplateMatcher.SIMILARITY_THRESHOLD,
        )

    except Exception as e:
        logger.exception("Template test failed")
        raise HTTPException(
            status_code=503,
            detail=f"Template testing unavailable: {e}",
        ) from e


@router.post(
    "/export",
    response_model=CustomTemplateExportResponse,
    response_model_exclude_unset=True,
    response_description="Exported template data",
    summary="Export templates for sharing",
    responses={
        200: {
            "description": "Export completed successfully",
        }
    },
)
def export_templates(
    request: CustomTemplateExportRequest | None = None,
) -> CustomTemplateExportResponse:
    """Export templates for sharing as a template pack.

    Args:
        request: Optional list of specific template IDs to export.
                 If null/empty, exports all templates.

    Returns:
        CustomTemplateExportResponse with template data for import
    """
    store = get_custom_template_store()

    template_ids = request.template_ids if request else None
    export_data = store.export_templates(template_ids)

    logger.info("Exported %d templates", export_data["template_count"])

    return CustomTemplateExportResponse(
        version=export_data["version"],
        export_date=export_data["export_date"],
        template_count=export_data["template_count"],
        templates=export_data["templates"],
    )


@router.post(
    "/import",
    response_model=CustomTemplateImportResponse,
    response_model_exclude_unset=True,
    response_description="Import results",
    summary="Import templates from export data",
    responses={
        200: {
            "description": "Import completed",
        },
        400: {
            "description": "Invalid import data",
            "model": ErrorResponse,
        },
    },
)
def import_templates(request: CustomTemplateImportRequest) -> CustomTemplateImportResponse:
    """Import templates from exported template pack.

    Args:
        request: Export data and import options

    Returns:
        CustomTemplateImportResponse with import results
    """
    if not request.data:
        raise HTTPException(
            status_code=400,
            detail="Import data is required",
        )

    if "templates" not in request.data:
        raise HTTPException(
            status_code=400,
            detail="Invalid import data: missing 'templates' field",
        )

    store = get_custom_template_store()
    result = store.import_templates(request.data, overwrite=request.overwrite)

    logger.info(
        "Imported %d templates (%d errors)",
        result["imported"],
        result["errors"],
    )

    return CustomTemplateImportResponse(
        imported=result["imported"],
        skipped=result["skipped"],
        errors=result["errors"],
        total_templates=result["total_templates"],
    )
