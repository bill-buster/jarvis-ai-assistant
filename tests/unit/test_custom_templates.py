"""Unit tests for custom templates functionality.

Tests the CustomTemplate model, CustomTemplateStore storage,
and the custom templates API endpoints.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from models.templates import (
    CustomTemplate,
    CustomTemplateStore,
    reset_custom_template_store,
)


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def temp_storage_path():
    """Create a temporary storage file location."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "custom_templates.json"


@pytest.fixture
def temp_store(temp_storage_path):
    """Create a CustomTemplateStore with a temporary storage path."""
    store = CustomTemplateStore(storage_path=temp_storage_path)
    yield store
    # Cleanup
    reset_custom_template_store()


class TestCustomTemplateModel:
    """Tests for CustomTemplate dataclass."""

    def test_default_values(self):
        """CustomTemplate has expected default values."""
        template = CustomTemplate()
        assert template.id != ""
        assert template.name == ""
        assert template.template_text == ""
        assert template.trigger_phrases == []
        assert template.category == "general"
        assert template.tags == []
        assert template.min_group_size is None
        assert template.max_group_size is None
        assert template.enabled is True
        assert template.usage_count == 0

    def test_custom_values(self):
        """CustomTemplate accepts custom values."""
        template = CustomTemplate(
            name="Test Template",
            template_text="Test response",
            trigger_phrases=["phrase1", "phrase2"],
            category="work",
            tags=["tag1", "tag2"],
            min_group_size=2,
            max_group_size=5,
            enabled=False,
        )
        assert template.name == "Test Template"
        assert template.template_text == "Test response"
        assert template.trigger_phrases == ["phrase1", "phrase2"]
        assert template.category == "work"
        assert template.tags == ["tag1", "tag2"]
        assert template.min_group_size == 2
        assert template.max_group_size == 5
        assert template.enabled is False

    def test_to_dict(self):
        """to_dict returns proper dictionary representation."""
        template = CustomTemplate(
            id="test-id",
            name="Test",
            template_text="Response",
            trigger_phrases=["phrase"],
            category="work",
        )
        d = template.to_dict()
        assert d["id"] == "test-id"
        assert d["name"] == "Test"
        assert d["template_text"] == "Response"
        assert d["trigger_phrases"] == ["phrase"]
        assert d["category"] == "work"

    def test_from_dict(self):
        """from_dict creates CustomTemplate from dictionary."""
        data = {
            "id": "test-id",
            "name": "Test",
            "template_text": "Response",
            "trigger_phrases": ["phrase1", "phrase2"],
            "category": "personal",
            "tags": ["tag"],
            "enabled": False,
        }
        template = CustomTemplate.from_dict(data)
        assert template.id == "test-id"
        assert template.name == "Test"
        assert template.template_text == "Response"
        assert template.trigger_phrases == ["phrase1", "phrase2"]
        assert template.category == "personal"
        assert template.tags == ["tag"]
        assert template.enabled is False

    def test_to_response_template(self):
        """to_response_template converts to ResponseTemplate."""
        template = CustomTemplate(
            id="test-id",
            name="Test",
            template_text="Test response",
            trigger_phrases=["hello", "hi"],
        )
        response_template = template.to_response_template()
        assert response_template.name == "custom_test-id"
        assert response_template.patterns == ["hello", "hi"]
        assert response_template.response == "Test response"


class TestCustomTemplateStore:
    """Tests for CustomTemplateStore."""

    def test_create_template(self, temp_store):
        """Creates a template successfully."""
        template = CustomTemplate(
            name="Test",
            template_text="Response",
            trigger_phrases=["phrase"],
        )
        created = temp_store.create(template)
        assert created.id != ""
        assert created.name == "Test"

    def test_get_template(self, temp_store):
        """Gets a template by ID."""
        template = CustomTemplate(name="Test", template_text="Response", trigger_phrases=["p"])
        created = temp_store.create(template)

        retrieved = temp_store.get(created.id)
        assert retrieved is not None
        assert retrieved.name == "Test"

    def test_get_nonexistent_template(self, temp_store):
        """Returns None for nonexistent template."""
        result = temp_store.get("nonexistent-id")
        assert result is None

    def test_list_all(self, temp_store):
        """Lists all templates."""
        for i in range(3):
            temp_store.create(
                CustomTemplate(
                    name=f"Template {i}",
                    template_text="Response",
                    trigger_phrases=["phrase"],
                )
            )

        templates = temp_store.list_all()
        assert len(templates) == 3

    def test_list_enabled(self, temp_store):
        """Lists only enabled templates."""
        temp_store.create(
            CustomTemplate(name="Enabled", template_text="R", trigger_phrases=["p"], enabled=True)
        )
        temp_store.create(
            CustomTemplate(name="Disabled", template_text="R", trigger_phrases=["p"], enabled=False)
        )

        templates = temp_store.list_enabled()
        assert len(templates) == 1
        assert templates[0].name == "Enabled"

    def test_list_by_category(self, temp_store):
        """Lists templates by category."""
        temp_store.create(
            CustomTemplate(name="Work", template_text="R", trigger_phrases=["p"], category="work")
        )
        temp_store.create(
            CustomTemplate(
                name="Personal", template_text="R", trigger_phrases=["p"], category="personal"
            )
        )

        work_templates = temp_store.list_by_category("work")
        assert len(work_templates) == 1
        assert work_templates[0].name == "Work"

    def test_list_by_tag(self, temp_store):
        """Lists templates by tag."""
        temp_store.create(
            CustomTemplate(
                name="Tagged", template_text="R", trigger_phrases=["p"], tags=["important"]
            )
        )
        temp_store.create(
            CustomTemplate(name="Untagged", template_text="R", trigger_phrases=["p"], tags=[])
        )

        tagged_templates = temp_store.list_by_tag("important")
        assert len(tagged_templates) == 1
        assert tagged_templates[0].name == "Tagged"

    def test_get_categories(self, temp_store):
        """Gets all unique categories."""
        temp_store.create(
            CustomTemplate(name="T1", template_text="R", trigger_phrases=["p"], category="work")
        )
        temp_store.create(
            CustomTemplate(name="T2", template_text="R", trigger_phrases=["p"], category="personal")
        )
        temp_store.create(
            CustomTemplate(name="T3", template_text="R", trigger_phrases=["p"], category="work")
        )

        categories = temp_store.get_categories()
        assert sorted(categories) == ["personal", "work"]

    def test_get_tags(self, temp_store):
        """Gets all unique tags."""
        temp_store.create(
            CustomTemplate(
                name="T1", template_text="R", trigger_phrases=["p"], tags=["tag1", "tag2"]
            )
        )
        temp_store.create(
            CustomTemplate(
                name="T2", template_text="R", trigger_phrases=["p"], tags=["tag2", "tag3"]
            )
        )

        tags = temp_store.get_tags()
        assert sorted(tags) == ["tag1", "tag2", "tag3"]

    def test_update_template(self, temp_store):
        """Updates a template."""
        template = CustomTemplate(name="Original", template_text="R", trigger_phrases=["p"])
        created = temp_store.create(template)

        updated = temp_store.update(created.id, {"name": "Updated"})
        assert updated is not None
        assert updated.name == "Updated"
        assert updated.updated_at != created.created_at

    def test_update_nonexistent_template(self, temp_store):
        """Returns None when updating nonexistent template."""
        result = temp_store.update("nonexistent-id", {"name": "Updated"})
        assert result is None

    def test_delete_template(self, temp_store):
        """Deletes a template."""
        template = CustomTemplate(name="ToDelete", template_text="R", trigger_phrases=["p"])
        created = temp_store.create(template)

        result = temp_store.delete(created.id)
        assert result is True
        assert temp_store.get(created.id) is None

    def test_delete_nonexistent_template(self, temp_store):
        """Returns False when deleting nonexistent template."""
        result = temp_store.delete("nonexistent-id")
        assert result is False

    def test_increment_usage(self, temp_store):
        """Increments usage count."""
        template = CustomTemplate(name="T", template_text="R", trigger_phrases=["p"])
        created = temp_store.create(template)
        assert created.usage_count == 0

        temp_store.increment_usage(created.id)
        temp_store.increment_usage(created.id)

        updated = temp_store.get(created.id)
        assert updated is not None
        assert updated.usage_count == 2

    def test_get_usage_stats(self, temp_store):
        """Gets usage statistics."""
        t1 = temp_store.create(
            CustomTemplate(
                name="T1", template_text="R", trigger_phrases=["p"], category="work", enabled=True
            )
        )
        t2 = temp_store.create(
            CustomTemplate(
                name="T2",
                template_text="R",
                trigger_phrases=["p"],
                category="personal",
                enabled=False,
            )
        )

        temp_store.increment_usage(t1.id)
        temp_store.increment_usage(t1.id)
        temp_store.increment_usage(t2.id)

        stats = temp_store.get_usage_stats()
        assert stats["total_templates"] == 2
        assert stats["enabled_templates"] == 1
        assert stats["total_usage"] == 3
        assert stats["usage_by_category"]["work"] == 2
        assert stats["usage_by_category"]["personal"] == 1

    def test_export_templates(self, temp_store):
        """Exports templates."""
        temp_store.create(
            CustomTemplate(name="T1", template_text="R1", trigger_phrases=["p1"])
        )
        temp_store.create(
            CustomTemplate(name="T2", template_text="R2", trigger_phrases=["p2"])
        )

        export_data = temp_store.export_templates()
        assert export_data["version"] == 1
        assert export_data["template_count"] == 2
        assert len(export_data["templates"]) == 2

    def test_export_specific_templates(self, temp_store):
        """Exports specific templates by ID."""
        t1 = temp_store.create(
            CustomTemplate(name="T1", template_text="R1", trigger_phrases=["p1"])
        )
        temp_store.create(CustomTemplate(name="T2", template_text="R2", trigger_phrases=["p2"]))

        export_data = temp_store.export_templates([t1.id])
        assert export_data["template_count"] == 1
        assert export_data["templates"][0]["name"] == "T1"

    def test_import_templates(self, temp_store):
        """Imports templates from export data."""
        import_data = {
            "version": 1,
            "templates": [
                {"name": "Imported1", "template_text": "R1", "trigger_phrases": ["p1"]},
                {"name": "Imported2", "template_text": "R2", "trigger_phrases": ["p2"]},
            ],
        }

        result = temp_store.import_templates(import_data)
        assert result["imported"] == 2
        assert result["errors"] == 0
        assert result["total_templates"] == 2

        templates = temp_store.list_all()
        names = [t.name for t in templates]
        assert "Imported1" in names
        assert "Imported2" in names

    def test_persistence(self, temp_storage_path):
        """Templates persist across store instances."""
        store1 = CustomTemplateStore(storage_path=temp_storage_path)
        store1.create(CustomTemplate(name="Persistent", template_text="R", trigger_phrases=["p"]))

        # Create new store instance with same path
        store2 = CustomTemplateStore(storage_path=temp_storage_path)
        templates = store2.list_all()
        assert len(templates) == 1
        assert templates[0].name == "Persistent"


class TestCustomTemplatesAPI:
    """Tests for custom templates API endpoints."""

    @patch("api.routers.custom_templates.get_custom_template_store")
    def test_list_templates_empty(self, mock_get_store, client):
        """Returns empty list when no templates exist."""
        mock_store = MagicMock()
        mock_store.list_all.return_value = []
        mock_store.get_categories.return_value = []
        mock_store.get_tags.return_value = []
        mock_get_store.return_value = mock_store

        response = client.get("/templates")

        assert response.status_code == 200
        data = response.json()
        assert data["templates"] == []
        assert data["total"] == 0

    @patch("api.routers.custom_templates.get_custom_template_store")
    def test_list_templates_with_templates(self, mock_get_store, client):
        """Returns list of templates."""
        template = CustomTemplate(
            id="test-id",
            name="Test",
            template_text="Response",
            trigger_phrases=["phrase"],
            category="work",
        )
        mock_store = MagicMock()
        mock_store.list_all.return_value = [template]
        mock_store.get_categories.return_value = ["work"]
        mock_store.get_tags.return_value = []
        mock_get_store.return_value = mock_store

        response = client.get("/templates")

        assert response.status_code == 200
        data = response.json()
        assert len(data["templates"]) == 1
        assert data["templates"][0]["name"] == "Test"
        assert data["categories"] == ["work"]

    @patch("api.routers.custom_templates.get_custom_template_store")
    def test_create_template(self, mock_get_store, client):
        """Creates a new template."""
        mock_store = MagicMock()
        mock_store.create.return_value = CustomTemplate(
            id="new-id",
            name="New Template",
            template_text="Response",
            trigger_phrases=["phrase"],
            category="general",
        )
        mock_get_store.return_value = mock_store

        response = client.post(
            "/templates",
            json={
                "name": "New Template",
                "template_text": "Response",
                "trigger_phrases": ["phrase"],
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "New Template"
        assert data["id"] == "new-id"

    @patch("api.routers.custom_templates.get_custom_template_store")
    def test_create_template_validates_trigger_phrases(self, mock_get_store, client):
        """Validates that trigger phrases are required."""
        response = client.post(
            "/templates",
            json={
                "name": "Test",
                "template_text": "Response",
                "trigger_phrases": [],
            },
        )

        # Pydantic validation returns 422 for min_length validation
        assert response.status_code == 422

    @patch("api.routers.custom_templates.get_custom_template_store")
    def test_get_template(self, mock_get_store, client):
        """Gets a specific template."""
        template = CustomTemplate(
            id="test-id",
            name="Test",
            template_text="Response",
            trigger_phrases=["phrase"],
        )
        mock_store = MagicMock()
        mock_store.get.return_value = template
        mock_get_store.return_value = mock_store

        response = client.get("/templates/test-id")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-id"
        assert data["name"] == "Test"

    @patch("api.routers.custom_templates.get_custom_template_store")
    def test_get_template_not_found(self, mock_get_store, client):
        """Returns 404 for nonexistent template."""
        mock_store = MagicMock()
        mock_store.get.return_value = None
        mock_get_store.return_value = mock_store

        response = client.get("/templates/nonexistent-id")

        assert response.status_code == 404

    @patch("api.routers.custom_templates.get_custom_template_store")
    def test_update_template(self, mock_get_store, client):
        """Updates a template."""
        original = CustomTemplate(
            id="test-id",
            name="Original",
            template_text="Response",
            trigger_phrases=["phrase"],
        )
        updated = CustomTemplate(
            id="test-id",
            name="Updated",
            template_text="Response",
            trigger_phrases=["phrase"],
        )
        mock_store = MagicMock()
        mock_store.get.return_value = original
        mock_store.update.return_value = updated
        mock_get_store.return_value = mock_store

        response = client.put(
            "/templates/test-id",
            json={"name": "Updated"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated"

    @patch("api.routers.custom_templates.get_custom_template_store")
    def test_delete_template(self, mock_get_store, client):
        """Deletes a template."""
        mock_store = MagicMock()
        mock_store.delete.return_value = True
        mock_get_store.return_value = mock_store

        response = client.delete("/templates/test-id")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"

    @patch("api.routers.custom_templates.get_custom_template_store")
    def test_delete_template_not_found(self, mock_get_store, client):
        """Returns 404 when deleting nonexistent template."""
        mock_store = MagicMock()
        mock_store.delete.return_value = False
        mock_get_store.return_value = mock_store

        response = client.delete("/templates/nonexistent-id")

        assert response.status_code == 404

    @patch("api.routers.custom_templates.get_custom_template_store")
    def test_get_usage_stats(self, mock_get_store, client):
        """Gets usage statistics."""
        mock_store = MagicMock()
        mock_store.get_usage_stats.return_value = {
            "total_templates": 5,
            "enabled_templates": 4,
            "total_usage": 100,
            "usage_by_category": {"work": 60, "personal": 40},
            "top_templates": [
                {"id": "t1", "name": "Template 1", "usage_count": 50},
            ],
        }
        mock_get_store.return_value = mock_store

        response = client.get("/templates/stats/usage")

        assert response.status_code == 200
        data = response.json()
        assert data["total_templates"] == 5
        assert data["total_usage"] == 100

    @patch("api.routers.custom_templates.get_custom_template_store")
    def test_export_templates(self, mock_get_store, client):
        """Exports templates."""
        mock_store = MagicMock()
        mock_store.export_templates.return_value = {
            "version": 1,
            "export_date": "2024-01-15T10:00:00",
            "template_count": 2,
            "templates": [{"name": "T1"}, {"name": "T2"}],
        }
        mock_get_store.return_value = mock_store

        response = client.post("/templates/export", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["version"] == 1
        assert data["template_count"] == 2

    @patch("api.routers.custom_templates.get_custom_template_store")
    def test_import_templates(self, mock_get_store, client):
        """Imports templates."""
        mock_store = MagicMock()
        mock_store.import_templates.return_value = {
            "imported": 3,
            "skipped": 0,
            "errors": 0,
            "total_templates": 3,
        }
        mock_get_store.return_value = mock_store

        response = client.post(
            "/templates/import",
            json={
                "data": {
                    "version": 1,
                    "templates": [
                        {"name": "T1", "template_text": "R1", "trigger_phrases": ["p1"]},
                    ],
                },
                "overwrite": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["imported"] == 3


class TestCustomTemplateSchemas:
    """Tests for custom template Pydantic schemas."""

    def test_custom_template_response_schema(self):
        """CustomTemplateResponse schema works correctly."""
        from api.schemas import CustomTemplateResponse

        response = CustomTemplateResponse(
            id="test-id",
            name="Test",
            template_text="Response",
            trigger_phrases=["phrase"],
            category="work",
            tags=["tag"],
            min_group_size=None,
            max_group_size=None,
            enabled=True,
            created_at="2024-01-15T10:00:00",
            updated_at="2024-01-15T10:00:00",
            usage_count=0,
        )
        assert response.id == "test-id"
        assert response.name == "Test"

    def test_custom_template_create_request_schema(self):
        """CustomTemplateCreateRequest schema works correctly."""
        from api.schemas import CustomTemplateCreateRequest

        request = CustomTemplateCreateRequest(
            name="Test",
            template_text="Response",
            trigger_phrases=["phrase1", "phrase2"],
        )
        assert request.name == "Test"
        assert len(request.trigger_phrases) == 2

    def test_custom_template_create_request_defaults(self):
        """CustomTemplateCreateRequest has correct defaults."""
        from api.schemas import CustomTemplateCreateRequest

        request = CustomTemplateCreateRequest(
            name="Test",
            template_text="Response",
            trigger_phrases=["phrase"],
        )
        assert request.category == "general"
        assert request.tags == []
        assert request.enabled is True

    def test_custom_template_update_request_optional(self):
        """CustomTemplateUpdateRequest has all optional fields."""
        from api.schemas import CustomTemplateUpdateRequest

        request = CustomTemplateUpdateRequest()
        assert request.name is None
        assert request.template_text is None
        assert request.trigger_phrases is None
        assert request.category is None
        assert request.tags is None
        assert request.enabled is None

    def test_custom_template_test_result_schema(self):
        """CustomTemplateTestResult schema works correctly."""
        from api.schemas import CustomTemplateTestResult

        result = CustomTemplateTestResult(
            input="test input",
            matched=True,
            best_match="trigger phrase",
            similarity=0.85,
        )
        assert result.matched is True
        assert result.similarity == 0.85

    def test_custom_template_export_response_schema(self):
        """CustomTemplateExportResponse schema works correctly."""
        from api.schemas import CustomTemplateExportResponse

        response = CustomTemplateExportResponse(
            version=1,
            export_date="2024-01-15T10:00:00",
            template_count=5,
            templates=[{"name": "T1"}, {"name": "T2"}],
        )
        assert response.version == 1
        assert response.template_count == 5

    def test_custom_template_import_response_schema(self):
        """CustomTemplateImportResponse schema works correctly."""
        from api.schemas import CustomTemplateImportResponse

        response = CustomTemplateImportResponse(
            imported=3,
            skipped=1,
            errors=0,
            total_templates=10,
        )
        assert response.imported == 3
        assert response.total_templates == 10
