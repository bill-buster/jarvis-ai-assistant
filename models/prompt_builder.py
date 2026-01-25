"""Prompt Builder for RAG context and few-shot formatting.

Formats prompts with injected context documents and example pairs
for optimal model performance.
"""

from contracts.models import GenerationRequest


class PromptBuilder:
    """Builder for formatting prompts with RAG context and few-shot examples.

    Structures prompts with clear sections for:
    - Relevant context documents (RAG)
    - Example input/output pairs (few-shot)
    - The user's actual task
    """

    CONTEXT_HEADER = "### Relevant Context:"
    EXAMPLES_HEADER = "### Examples:"
    TASK_HEADER = "### Your Task:"
    SECTION_SEPARATOR = "\n\n"
    DOCUMENT_SEPARATOR = "\n---\n"

    def build(self, request: GenerationRequest) -> str:
        """Build a formatted prompt from a generation request.

        Args:
            request: The generation request with prompt, context, and examples

        Returns:
            Formatted prompt string ready for model input
        """
        sections = []

        # Add context section if documents provided
        if request.context_documents:
            context = self._format_context(request.context_documents)
            sections.append(context)

        # Add examples section if examples provided
        if request.few_shot_examples:
            examples = self._format_examples(request.few_shot_examples)
            sections.append(examples)

        # Add the actual task
        task = self._format_task(request.prompt)
        sections.append(task)

        return self.SECTION_SEPARATOR.join(sections)

    def _format_context(self, documents: list[str]) -> str:
        """Format context documents section.

        Args:
            documents: List of context document strings

        Returns:
            Formatted context section
        """
        formatted_docs = self.DOCUMENT_SEPARATOR.join(doc.strip() for doc in documents)
        return f"{self.CONTEXT_HEADER}\n{formatted_docs}"

    def _format_examples(self, examples: list[tuple[str, str]]) -> str:
        """Format few-shot examples section.

        Args:
            examples: List of (input, output) example pairs

        Returns:
            Formatted examples section
        """
        formatted_examples = []
        for input_text, output_text in examples:
            example = f"Input: {input_text.strip()}\nOutput: {output_text.strip()}"
            formatted_examples.append(example)

        examples_text = "\n\n".join(formatted_examples)
        return f"{self.EXAMPLES_HEADER}\n{examples_text}"

    def _format_task(self, prompt: str) -> str:
        """Format the user's task section.

        Args:
            prompt: The user's prompt/task

        Returns:
            Formatted task section
        """
        return f"{self.TASK_HEADER}\n{prompt.strip()}"
