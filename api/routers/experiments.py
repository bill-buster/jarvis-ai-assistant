"""A/B Testing experiment management API endpoints.

Provides endpoints for managing experiments, recording outcomes, and
viewing experiment results with statistical analysis.

Endpoints:
- GET /experiments - List all active experiments
- GET /experiments/{name} - Get a specific experiment
- GET /experiments/{name}/results - Get experiment results with statistics
- POST /experiments/{name}/record - Record an experiment outcome
- POST /experiments - Create a new experiment
- PUT /experiments/{name} - Update an experiment
- DELETE /experiments/{name} - Delete an experiment
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from jarvis.experiments import (
    ExperimentManager,
    ExperimentResults,
    UserAction,
    get_experiment_manager,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/experiments", tags=["experiments"])


# =============================================================================
# Request/Response Schemas
# =============================================================================


class VariantConfigResponse(BaseModel):
    """Response model for a variant configuration."""

    id: str = Field(..., description="Unique variant identifier")
    weight: int = Field(..., description="Allocation weight (0-100)")
    config: dict[str, Any] = Field(default_factory=dict, description="Variant configuration")


class ExperimentResponse(BaseModel):
    """Response model for an experiment."""

    name: str = Field(..., description="Unique experiment name")
    enabled: bool = Field(..., description="Whether the experiment is active")
    description: str = Field(default="", description="Experiment description")
    created_at: str = Field(default="", description="Creation timestamp")
    variants: list[VariantConfigResponse] = Field(..., description="List of variants")


class VariantResultsResponse(BaseModel):
    """Response model for variant results."""

    variant_id: str = Field(..., description="Variant identifier")
    total_impressions: int = Field(..., description="Total times variant was shown")
    sent_unchanged: int = Field(..., description="Times user sent without editing")
    sent_edited: int = Field(..., description="Times user edited before sending")
    dismissed: int = Field(..., description="Times user dismissed")
    regenerated: int = Field(..., description="Times user regenerated")
    conversion_rate: float = Field(..., description="Percentage of sent_unchanged")


class ExperimentResultsResponse(BaseModel):
    """Response model for experiment results."""

    experiment_name: str = Field(..., description="Experiment name")
    variants: list[VariantResultsResponse] = Field(..., description="Results per variant")
    total_outcomes: int = Field(..., description="Total recorded outcomes")
    winner: str | None = Field(None, description="Winning variant ID (highest conversion)")
    is_significant: bool = Field(..., description="Whether results are statistically significant")
    p_value: float | None = Field(None, description="P-value from chi-squared test")


class RecordOutcomeRequest(BaseModel):
    """Request to record an experiment outcome."""

    variant_id: str = Field(..., description="ID of the variant that was shown")
    contact_id: str = Field(..., description="Identifier for the contact")
    action: str = Field(
        ...,
        description="User action: sent_unchanged, sent_edited, dismissed, regenerated",
    )


class RecordOutcomeResponse(BaseModel):
    """Response after recording an outcome."""

    success: bool = Field(..., description="Whether the outcome was recorded")
    experiment_name: str = Field(..., description="Experiment name")
    variant_id: str = Field(..., description="Variant ID")
    action: str = Field(..., description="Recorded action")


class CreateExperimentRequest(BaseModel):
    """Request to create a new experiment."""

    name: str = Field(..., min_length=1, max_length=100, description="Unique experiment name")
    description: str = Field(default="", description="Experiment description")
    enabled: bool = Field(default=True, description="Whether to enable the experiment")
    variants: list[dict[str, Any]] = Field(
        ...,
        min_length=2,
        description="List of variants with id, weight, and config",
    )


class UpdateExperimentRequest(BaseModel):
    """Request to update an experiment."""

    enabled: bool | None = Field(None, description="New enabled state")


class ExperimentListResponse(BaseModel):
    """Response with list of experiments."""

    experiments: list[ExperimentResponse] = Field(..., description="List of experiments")
    total: int = Field(..., description="Total number of experiments")


# =============================================================================
# Helper Functions
# =============================================================================


def _experiment_to_response(exp: Any) -> ExperimentResponse:
    """Convert Experiment to ExperimentResponse."""
    return ExperimentResponse(
        name=exp.name,
        enabled=exp.enabled,
        description=exp.description,
        created_at=exp.created_at,
        variants=[
            VariantConfigResponse(id=v.id, weight=v.weight, config=v.config)
            for v in exp.variants
        ],
    )


def _results_to_response(results: ExperimentResults) -> ExperimentResultsResponse:
    """Convert ExperimentResults to ExperimentResultsResponse."""
    return ExperimentResultsResponse(
        experiment_name=results.experiment_name,
        variants=[
            VariantResultsResponse(
                variant_id=v.variant_id,
                total_impressions=v.total_impressions,
                sent_unchanged=v.sent_unchanged,
                sent_edited=v.sent_edited,
                dismissed=v.dismissed,
                regenerated=v.regenerated,
                conversion_rate=v.conversion_rate,
            )
            for v in results.variants
        ],
        total_outcomes=results.total_outcomes,
        winner=results.winner,
        is_significant=results.is_significant,
        p_value=results.p_value,
    )


def _get_manager() -> ExperimentManager:
    """Get the experiment manager singleton."""
    return get_experiment_manager()


# =============================================================================
# Endpoints
# =============================================================================


@router.get(
    "",
    response_model=ExperimentListResponse,
    summary="List all experiments",
    responses={
        200: {"description": "List of all experiments"},
    },
)
def list_experiments(active_only: bool = False) -> ExperimentListResponse:
    """List all configured experiments.

    Returns a list of all experiments with their configurations.
    Optionally filter to show only active (enabled) experiments.

    Args:
        active_only: If true, only return enabled experiments.

    Returns:
        ExperimentListResponse with list of experiments.
    """
    manager = _get_manager()

    if active_only:
        experiments = manager.get_active_experiments()
    else:
        experiments = manager.get_experiments()

    return ExperimentListResponse(
        experiments=[_experiment_to_response(exp) for exp in experiments],
        total=len(experiments),
    )


@router.get(
    "/{name}",
    response_model=ExperimentResponse,
    summary="Get experiment details",
    responses={
        200: {"description": "Experiment details"},
        404: {"description": "Experiment not found"},
    },
)
def get_experiment(name: str) -> ExperimentResponse:
    """Get details of a specific experiment.

    Args:
        name: The experiment name.

    Returns:
        ExperimentResponse with experiment details.

    Raises:
        HTTPException 404: Experiment not found.
    """
    manager = _get_manager()
    experiment = manager.get_experiment(name)

    if not experiment:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment not found: {name}",
        )

    return _experiment_to_response(experiment)


@router.get(
    "/{name}/results",
    response_model=ExperimentResultsResponse,
    summary="Get experiment results",
    responses={
        200: {"description": "Experiment results with statistics"},
        404: {"description": "Experiment not found"},
    },
)
def get_experiment_results(name: str) -> ExperimentResultsResponse:
    """Get results and statistics for an experiment.

    Returns win rates per variant, statistical significance indicators,
    and p-value from chi-squared test (for two-variant experiments).

    Args:
        name: The experiment name.

    Returns:
        ExperimentResultsResponse with results and statistics.

    Raises:
        HTTPException 404: Experiment not found.
    """
    manager = _get_manager()
    results = manager.calculate_results(name)

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment not found: {name}",
        )

    return _results_to_response(results)


@router.post(
    "/{name}/record",
    response_model=RecordOutcomeResponse,
    summary="Record experiment outcome",
    responses={
        200: {"description": "Outcome recorded successfully"},
        400: {"description": "Invalid action"},
        404: {"description": "Experiment not found"},
    },
)
def record_outcome(name: str, request: RecordOutcomeRequest) -> RecordOutcomeResponse:
    """Record an outcome for an experiment.

    Records the user's action (sent_unchanged, sent_edited, dismissed, regenerated)
    for tracking conversion rates and calculating statistical significance.

    Args:
        name: The experiment name.
        request: RecordOutcomeRequest with variant_id, contact_id, and action.

    Returns:
        RecordOutcomeResponse confirming the recorded outcome.

    Raises:
        HTTPException 400: Invalid action.
        HTTPException 404: Experiment not found.
    """
    manager = _get_manager()
    experiment = manager.get_experiment(name)

    if not experiment:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment not found: {name}",
        )

    # Validate action
    try:
        action = UserAction(request.action)
    except ValueError:
        valid_actions = [a.value for a in UserAction]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action: {request.action}. Valid actions: {valid_actions}",
        )

    success = manager.record_outcome(
        experiment_name=name,
        variant_id=request.variant_id,
        contact_id=request.contact_id,
        action=action,
    )

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to record outcome",
        )

    return RecordOutcomeResponse(
        success=True,
        experiment_name=name,
        variant_id=request.variant_id,
        action=request.action,
    )


@router.post(
    "",
    response_model=ExperimentResponse,
    summary="Create new experiment",
    status_code=201,
    responses={
        201: {"description": "Experiment created"},
        400: {"description": "Invalid configuration"},
        409: {"description": "Experiment already exists"},
    },
)
def create_experiment(request: CreateExperimentRequest) -> ExperimentResponse:
    """Create a new A/B testing experiment.

    Each experiment needs at least two variants with allocation weights
    that should sum to 100.

    **Example Request:**
    ```json
    {
        "name": "reply_tone_test",
        "description": "Test casual vs professional reply tones",
        "enabled": true,
        "variants": [
            {
                "id": "control",
                "weight": 50,
                "config": {
                    "system_prompt": "You are a helpful assistant.",
                    "temperature": 0.7
                }
            },
            {
                "id": "treatment",
                "weight": 50,
                "config": {
                    "system_prompt": "You are a friendly, casual assistant.",
                    "temperature": 0.8
                }
            }
        ]
    }
    ```

    Args:
        request: CreateExperimentRequest with experiment configuration.

    Returns:
        ExperimentResponse with the created experiment.

    Raises:
        HTTPException 400: Invalid configuration.
        HTTPException 409: Experiment already exists.
    """
    manager = _get_manager()

    # Check if experiment already exists
    existing = manager.get_experiment(request.name)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Experiment already exists: {request.name}",
        )

    # Validate variants
    if len(request.variants) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least two variants are required",
        )

    # Validate weights sum to 100
    total_weight = sum(v.get("weight", 50) for v in request.variants)
    if total_weight != 100:
        raise HTTPException(
            status_code=400,
            detail=f"Variant weights must sum to 100 (got {total_weight})",
        )

    # Validate each variant has an ID
    for i, variant in enumerate(request.variants):
        if not variant.get("id"):
            raise HTTPException(
                status_code=400,
                detail=f"Variant at index {i} missing required 'id' field",
            )

    experiment = manager.create_experiment(
        name=request.name,
        variants=request.variants,
        description=request.description,
        enabled=request.enabled,
    )

    return _experiment_to_response(experiment)


@router.put(
    "/{name}",
    response_model=ExperimentResponse,
    summary="Update experiment",
    responses={
        200: {"description": "Experiment updated"},
        404: {"description": "Experiment not found"},
    },
)
def update_experiment(name: str, request: UpdateExperimentRequest) -> ExperimentResponse:
    """Update an existing experiment.

    Currently supports enabling/disabling experiments.

    Args:
        name: The experiment name.
        request: UpdateExperimentRequest with new values.

    Returns:
        ExperimentResponse with updated experiment.

    Raises:
        HTTPException 404: Experiment not found.
    """
    manager = _get_manager()
    experiment = manager.get_experiment(name)

    if not experiment:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment not found: {name}",
        )

    success = manager.update_experiment(name, enabled=request.enabled)
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to update experiment",
        )

    # Reload to get updated state
    experiment = manager.get_experiment(name)
    return _experiment_to_response(experiment)


@router.delete(
    "/{name}",
    summary="Delete experiment",
    responses={
        200: {"description": "Experiment deleted"},
        404: {"description": "Experiment not found"},
    },
)
def delete_experiment(name: str) -> dict[str, Any]:
    """Delete an experiment.

    Warning: This will permanently delete the experiment configuration.
    Recorded outcomes are not deleted.

    Args:
        name: The experiment name.

    Returns:
        Confirmation message.

    Raises:
        HTTPException 404: Experiment not found.
    """
    manager = _get_manager()
    experiment = manager.get_experiment(name)

    if not experiment:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment not found: {name}",
        )

    success = manager.delete_experiment(name)
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to delete experiment",
        )

    return {"status": "deleted", "experiment_name": name}


@router.delete(
    "/{name}/outcomes",
    summary="Clear experiment outcomes",
    responses={
        200: {"description": "Outcomes cleared"},
        404: {"description": "Experiment not found"},
    },
)
def clear_experiment_outcomes(name: str) -> dict[str, Any]:
    """Clear all recorded outcomes for an experiment.

    Use this to reset experiment data when starting a new test cycle.

    Args:
        name: The experiment name.

    Returns:
        Confirmation message.

    Raises:
        HTTPException 404: Experiment not found.
    """
    manager = _get_manager()
    experiment = manager.get_experiment(name)

    if not experiment:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment not found: {name}",
        )

    success = manager.clear_outcomes(name)
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to clear outcomes",
        )

    return {"status": "cleared", "experiment_name": name}


@router.get(
    "/{name}/variant",
    response_model=VariantConfigResponse | None,
    summary="Get variant for contact",
    responses={
        200: {"description": "Assigned variant"},
        404: {"description": "Experiment not found or disabled"},
    },
)
def get_variant_for_contact(name: str, contact_id: str) -> VariantConfigResponse | None:
    """Get the variant assignment for a specific contact.

    Uses consistent hashing to ensure the same contact always gets
    the same variant for a given experiment.

    Args:
        name: The experiment name.
        contact_id: Identifier for the contact (phone, email, etc.).

    Returns:
        VariantConfigResponse with the assigned variant, or None if disabled.

    Raises:
        HTTPException 404: Experiment not found.
    """
    manager = _get_manager()
    experiment = manager.get_experiment(name)

    if not experiment:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment not found: {name}",
        )

    variant = manager.get_variant(name, contact_id)
    if not variant:
        return None

    return VariantConfigResponse(id=variant.id, weight=variant.weight, config=variant.config)
